'use strict';

// canvas to draw input image
var canvas_draw, canvas_draw_ctx;

var webdnn_runner = null;
var webdnn_input_view, webdnn_output_view;
var dnn_calculating = false;

async function initialize() {
  webdnn_runner = await WebDNN.load('./webdnn_graph_descriptor', {backendOrder: 'webgl'});
  webdnn_input_view = webdnn_runner.getInputViews()[0];
  webdnn_output_view = webdnn_runner.getOutputViews()[0];
  console.log('loaded webdnn model');
}

// called when the input image is updated
async function calculate() {
  console.log('calc');
  if (dnn_calculating) {
    return false;
  }
  dnn_calculating = true;

  // set input data by scaling and flattening the image in canvas
  // https://mil-tokyo.github.io/webdnn/docs/api_reference/descriptor-runner/modules/webdnn_image.html
  webdnn_input_view.set(await WebDNN.Image.getImageArray(canvas_draw, {
    dstH: 28, dstW: 28,
    order: WebDNN.Image.Order.HWC,
    color: WebDNN.Image.Color.GREY,
    bias: [0, 0, 0],
    scale: [255, 255, 255]
  }));

  // run the DNN
  await webdnn_runner.run();

  // get the result
  var probabilities = webdnn_output_view.toActual();
  // display the result
  var result_html = '';
  for (var i = 0; i < 10; i++) {
    var probability_percent = Math.floor(probabilities[i] * 100);
    result_html += '' + i + ': ' + probability_percent + '%<br>';
  }
  document.getElementById('result').innerHTML = result_html;

  dnn_calculating = false;
}

// mouse drawing related variables
var draw_mouse_down = false;
var point_radius = 20;

// mouse drawing
function draw_point(x, y) {
  canvas_draw_ctx.fillStyle = 'white';
  canvas_draw_ctx.beginPath();
  canvas_draw_ctx.arc(x, y, point_radius, 0, Math.PI * 2, false);
  canvas_draw_ctx.fill();
}

function draw_clear() {
  // clearRect fills with TRANSPARENT black; not suitable
  canvas_draw_ctx.fillStyle = 'black';
  canvas_draw_ctx.fillRect(0, 0, canvas_draw.width, canvas_draw.height);
}

var update_timer_id = null;
function reserve_update() {
  if (update_timer_id) {
    clearTimeout(update_timer_id);
  }
  update_timer_id = setTimeout(function() {
    update_timer_id = null;
    calculate().then(()=>{});
  }, 100);
}

window.addEventListener('load', function(){
  canvas_draw = document.getElementById('draw');
  canvas_draw_ctx = canvas_draw.getContext('2d');
  draw_clear();

  // mouse drawing events
  canvas_draw.addEventListener('mousedown', function (e) {
    draw_mouse_down = true;
    var rect = e.target.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    draw_point(x, y);
    reserve_update();
  });
  canvas_draw.addEventListener('mousemove', function (e) {
    if (draw_mouse_down) {
      var rect = e.target.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      draw_point(x, y);
      reserve_update();
    }
  });
  canvas_draw.addEventListener('mouseup', function (e) {
    draw_mouse_down = false;
  });

  document.getElementById('reset').addEventListener('click', function (e) {
    draw_clear();
    reserve_update();
  });

  initialize().then(()=>{});
});
