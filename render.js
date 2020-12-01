let w,h;

let canvas,context;

let renderer;
let render;

let uniforms;

let hash; 
let texture;

let mouse;

let cam,scene,geometry,mesh,material;

function init() {

    canvas  = $('#canvas')[0];
    context = canvas.getContext('webgl2',{ antialias:false });

    w = window.innerWidth;
    h = window.innerHeight;

    renderer = new THREE.WebGLRenderer(
    { canvas:canvas,context:context });

    hash = new Math.seedrandom();

    mouse = new THREE.Vector2(0.0); 

    cam = new THREE.PerspectiveCamera(0.0,w/h,0.1,1.0);

    scene = new THREE.Scene();
    geometry = new THREE.PlaneBufferGeometry(2,2);

    updateTex(w,h);

    uniforms = {

        "time"       : { value : 1.0 },
        "resolution" : new THREE.Uniform(new THREE.Vector2(w,h)),
        "mouse"      : new THREE.Uniform(new THREE.Vector2()),
        "seed"       : { value: hash.int32() },
        "tex"        : { type:"t", value: texture }

    };   

}

init();

ShaderLoader("render.vert","vortex.frag",

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
    
        uniforms["time"  ].value = performance.now();
        uniforms["mouse" ].value = mouse;
    
        renderer.render(scene,cam);

        } 
       
    render();

    }
) 

function updateTex(w,h) {

    let size = w*h;
    let data = new Uint8Array(3 * size);

        for(let i = 0; i < size; i++) {
                             
                let s =  i * 3;
                let r = 255 * hash();        

                data[s]     = Math.floor(r);
                data[s+1]   = Math.floor(r);
                data[s+2]   = Math.floor(r);   
                
            }
               
     texture = new THREE.DataTexture(data,w,h,THREE.RGBFormat);
     texture.magFilter = THREE.LinearFilter;
  
}

$('#canvas').keydown(function(event) {
 
    if(event.which == 82) {
        event.preventDefault(); 
        window.location.reload();
    }
});

window.addEventListener('mousemove',onMouseMove,false);

function onMouseMove(event) {
    mouse.x = (event.clientX / w) * 2.0 - 1.0; 
    mouse.y = -(event.clientY / h) * 2.0 + 1.0;
}
