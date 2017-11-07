'use strict';

// Simple paiting feature for canvas

window.addEventListener('load', function () {
  var canvas_draw = document.getElementById('draw');
  var canvas_draw_ctx = canvas_draw.getContext('2d');
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
  var dnn_calculating = false;
  function reserve_update() {
    if (update_timer_id) {
      clearTimeout(update_timer_id);
    }
    update_timer_id = setTimeout(function () {
      update_timer_id = null;
      if (!dnn_calculating) {
        dnn_calculating = true;
        calculate().then(() => {
          dnn_calculating = false;
        }).catch((reason) => {
          console.error('Failed DNN calculation!', reason);
        });
      } else {
        // DNN is now calculating, so avoid another run
      }
    }, 100);
  }

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

});
