let model = null;
const loadModel = async () => {
    model = await tf.loadLayersModel('./model.json');
}

loadModel();

const CANVAS_SIZE = 28;

const canvas = document.getElementById('canvas');

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

const ctx = canvas.getContext('2d');

ctx.scale(1,1);

canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

// const scale = CANVAS_SIZE / CANVAS_SIZE;
// ctx.setTransform(scale, 0, 0, scale, 0, 0);

let isDrawing = false;
const strokeWidth = 10;

const draw = (e) => {
    if (!isDrawing)
        return; 

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / (rect.width / 28));
    const y = Math.floor((e.clientY - rect.top) / (rect.height / 28));
    
    ctx.fillStyle = 'black';
    ctx.fillRect(x, y, 1, 1);
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
});

canvas.addEventListener('mouseup', (e) => {
    isDrawing = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);

const processCanvasImage = (imageData) => {
    const data = imageData.data;
    const grayscaleData = [];

    for (let i = 0; i < data.length; i += 4) {
        const grayscalePixel = 255 - Math.floor((data[i] + data[i + 1] + data[i + 2]) / 3);
        grayscaleData.push(grayscalePixel);
    }

    return tf.tensor4d(grayscaleData, [1, 28, 28, 1]);
}

const getPrediction = (data) => {
    const predictionTensor = model.predict(data);
    const value = predictionTensor.argMax(-1).dataSync()[0];

    document.getElementsByClassName("outputNode")[value].classList += " predicted"
    predictButton.removeAttribute("disabled")
}

const predict = () => {
    predictButton.setAttribute("disabled", true);
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const processedImage = processCanvasImage(imageData);

    getPrediction(processedImage);
}

const resetOutputNodes = () => {
    const outputNodes = document.getElementsByClassName("outputNode");
    for (let i = 0; i < outputNodes.length; i++) {
        outputNodes[i].classList = "outputNode";
    }
}

const predictButton = document.getElementById('predictButton');
predictButton.addEventListener('click', () => {
    resetOutputNodes();
    predict();
});

const clearButton = document.getElementById('clearButton');
clearButton.addEventListener('click', () => {
    resetOutputNodes();

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
});
