'use strict';

var webdnn_runner = null;
var webdnn_input_view, webdnn_output_view;

// This function is called when the page is loaded
async function initialize() {
  console.log('Beginning of initialize()');
  // Load WebDNN model
  webdnn_runner = await WebDNN.load('./webdnn_graph_descriptor');
  // Get reference for input / output variable of the model
  // There can be multiple variables for input / output, but there is only one for this model
  webdnn_input_view = webdnn_runner.getInputViews()[0];
  webdnn_output_view = webdnn_runner.getOutputViews()[0];
  console.log('End of initialize()');
}

// This function is called when the input image is updated
async function calculate() {
  console.log('Beginning of calculate()');

  // set input data by scaling and flattening the image in canvas
  // https://mil-tokyo.github.io/webdnn/docs/api_reference/descriptor-runner/modules/webdnn_image.html
  var canvas_draw = document.getElementById('draw'); // canvas object that contains input image
  webdnn_input_view.set(await WebDNN.Image.getImageArray(canvas_draw, {
    dstH: 28, dstW: 28,
    order: WebDNN.Image.Order.HWC, // for Keras
    // order: WebDNN.Image.Order.CHW, // for Chainer (but the same as HWC because channel=1 in MNIST)
    color: WebDNN.Image.Color.GREY,
    bias: [0, 0, 0],
    scale: [255, 255, 255]
  }));

  // run the DNN
  await webdnn_runner.run();

  // get the result
  var probabilities = webdnn_output_view.toActual();
  // 'probabilities' is array containing each label's probability (0.0 to 1.0)
  // display the result
  var result_html = '';
  for (var i = 0; i < 10; i++) {
    var probability_percent = Math.floor(probabilities[i] * 100);
    result_html += '' + i + ': ' + probability_percent + '%<br>'; // <br> makes new line
  }
  document.getElementById('result').innerHTML = result_html; // display result in 'result' element

  console.log('End of calculate()');
}

window.addEventListener('load', function () {
  initialize().then(() => { }).catch((reason) => {
    console.error('Failed to initialize', reason);
  });
});
