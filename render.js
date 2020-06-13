let w,h;

let canvas,context;

let renderer;
let render;
let uniforms;

let reset;

let nhash,hash;  
let texture;

let mouse_pressed,mouse_held,mouse;

let controls;

let cam,scene,geometry,mesh,mat;

let cam_target;

const phi = (1. + Math.sqrt(5.)) / 2.;

function init() {

    canvas  = $('#canvas')[0];
    context = canvas.getContext('webgl2',{ antialias:false });

    w = window.innerWidth;
    h = window.innerHeight;

    renderer = new THREE.WebGLRenderer({canvas:canvas,context:context});

    cam = new THREE.PerspectiveCamera(45.,w/h,0.0,1.0);

    nhash = new Math.seedrandom();
    hash = nhash();

    mouse = new THREE.Vector2(0.0); 
    mouse_pressed = 0;
    mouse_held = 0;
    swipe_dir = 0;

    cam.position.set(0.25,1.25,.75); 
    cam_target  = new THREE.Vector3(0.0);

    controls = new THREE.OrbitControls(cam,canvas);

        controls.minDistance = 0.0;
        controls.maxDistance = 2.0;
        controls.target = cam_target;
        controls.enableDamping = true;
        controls.maxPolarAngle = 1. / phi;
        controls.enablePan = false; 
        controls.enabled = true;

    scene = new THREE.Scene();
    geometry = new THREE.PlaneBufferGeometry(2,2);

    uniforms = {

        "u_time"                : { value : 1.0 },
        "u_resolution"          : new THREE.Uniform(new THREE.Vector2(w,h)),
        "u_mouse"               : new THREE.Uniform(new THREE.Vector2()),
        "u_mouse_pressed"       : { value : mouse_pressed },
        "u_swipe_dir"           : { value : swipe_dir }, 
        "u_cam_target"          : new THREE.Uniform(new THREE.Vector3(cam_target)),
        "u_hash"                : { value: hash },
        "u_tex"                 : { type:"t", value: texture }

    };   

}

init();

ShaderLoader("render.vert","render.frag",

    function(vertex,fragment) {

        material = new THREE.ShaderMaterial({

            uniforms : uniforms,
            vertexShader : vertex,
            fragmentShader : fragment

        });

        mesh = new THREE.Mesh(geometry,material);

        scene.add(mesh);
        
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(w,h);

        render = function(timestamp) {

        requestAnimationFrame(render);
    
        uniforms["u_time"                ].value = performance.now();
        uniforms["u_mouse"               ].value = mouse;
        uniforms["u_mouse_pressed"       ].value = mouse_pressed;
        uniforms["u_swipe_dir"           ].value = swipe_dir;
        uniforms["u_cam_target"          ].value = cam_target;
        uniforms["u_hash"                ].value = hash;
        uniforms["u_tex"                 ].value = texture;       

        if(cam.position.lengthSq() < .15) {
            controls.maxPolarAngle = Math.PI;
        } else { 
            controls.maxPolarAngle = 1. / phi;
        }

        controls.update();
        renderer.render(scene,cam);

        } 
       
    render();

    }
) 

function updateTex() {

    let size = 16 * 16;
    let data = new Uint8Array(3 * size);

        for(let i = 0; i < size; i++) {
                             
                let s =  i * 3;

                data[s]     = Math.floor( 255 * nhash()    );
                data[s+1]   = Math.floor( 255 * nhash()    );
                data[s+2]   = Math.floor( 255 * nhash()    );   
                
            }
               

     texture = new THREE.DataTexture(tex,16,16,THREE.RGBFormat);
     texture.magFilter = THREE.LinearFilter;
  
}

$('#canvas').keydown(function(event) {
 
    if(event.which == 37) {
        event.preventDefault(); 
   
    }

    if(event.which == 38 ) {
        event.preventDefault();

    }
    
    if(event.which == 39 ) {
        event.preventDefault();

    }

    if(event.which == 40 ) {
        event.preventDefault();

    }

});

$('#canvas').mousedown(function() { 
 
    mouse_pressed = true;
   
    reset = setTimeout(function() {
    mouse_held = true; 

    },2500);


});

$('#canvas').mouseup(function() {
    
    mouse_pressed = false;    
    mouse_held = false;
    
    if(reset) {
        clearTimeout(reset);
    };

});        

window.addEventListener('mousemove',onMouseMove,false);

function onMouseMove(event) {
    mouse.x = (event.clientX / w) * 2.0 - 1.0; 
    mouse.y = -(event.clientY / h) * 2.0 + 1.0;
}
