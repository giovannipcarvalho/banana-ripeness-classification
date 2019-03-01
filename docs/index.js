const MODEL_PATH = 'model_js/model.json';
const IMAGE_SIZE = 224;
const CLASS_LABELS = {
    0: 'green',
    1: 'overripe',
    2: 'ripe'
}

const selected_image = document.getElementById('selected-image');
const image_selector = document.getElementById('image-selector');
const status_message = document.getElementById('status');
const predict_result = document.getElementById('prediction-result');

const status = msg => status_message.innerText = msg;

let model;
const modelDemo = async () => {
    status('Loading model...');
    // load model
    model = await tf.loadLayersModel(MODEL_PATH);
    // warmup
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    status('');

    // predict sample image
    if (selected_image.complete && selected_image.naturalHeight !== 0) {
        predict(selected_image);
    } else {
        selected_image.onload = () => {
            predict_button(selected_image);
        }
    }
};

async function predict(image) {
    // clean text before predicting
    predict_result.innerText = '';

    status('Predicting...');
    const startTime = performance.now();
    const logits = tf.tidy(() => {
        const img = tf.browser.fromPixels(image).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = img.sub(offset).div(offset);
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        return model.predict(batched);
    });

    showResult(logits);
    
    const totalTime = performance.now() - startTime;
    status(`Done in ${Math.floor(totalTime)}ms.`);
}

async function showResult(logits) {
    const message = {
        0: 'This banana looks a little green! Wait a little longer before eating it.',
        1: 'This banana looks rotten or overripe. I would not eat it if I were you.',
        2: 'Yum! Looks ripe to me!'
    }

    const values = await logits.data();
    console.log(values);

    const class_idx = logits.argMax(1).dataSync();
    predict_result.innerText = message[class_idx];
}

image_selector.addEventListener('change', evt => {
    let files = evt.target.files;

    for (let i = 0, f; f = files[i]; ++i) {
        if (!f.type.match('image.*')) { continue; }

        let reader = new FileReader();
        reader.onload = e => {
            selected_image.src = e.target.result;
            selected_image.width = IMAGE_SIZE;
            selected_image.height = IMAGE_SIZE;
            selected_image.onload = () => predict(selected_image);
        };

        reader.readAsDataURL(f);
        break;
    }
});

modelDemo();
