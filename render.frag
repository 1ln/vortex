#version 300 es

// dolson,2019

precision mediump float;
precision mediump sampler2D;

out vec4 out_FragColor;
varying vec2 uVu; 

uniform float u_hash;

uniform vec2 u_mouse;
uniform int u_mouse_pressed;
uniform int u_swipe_dir;

uniform vec2 u_resolution;

uniform vec3 u_cam_target;

uniform float u_time;

uniform sampler2D u_noise_tex;

const float E   =  2.7182818;
const float PI  =  radians(180.0); 
const float PHI =  (1.0 + sqrt(5.0)) / 2.0;

//15551*89491 = 1391674541
float hash(float p) {
    uvec2 n = uint(int(p)) * uvec2(1391674541U,2531151992.0*u_hash);
    uint h = (n.x ^ n.y) * 1391674541U;
    return float(h) * (1.0/float(0xffffffffU));
}
 
float noise(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
                   mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
               mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
                   mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}


float fractal(vec3 x,int octaves,float h) {

    float t = 0.0;

    float g = exp2(-h); 

    float a = 0.5;
    float f = 1.0;

    for(int i = 0; i < octaves; i++) {
 
    t += a * noise(f * x); 
    f *= 2.0; 
    a *=  g;  
    
    }    

    return t;
}

float sin3(vec3 p,float h) {
    
    return sin(p.x*h) * sin(p.y*h) * sin(p.z*h);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    
    return a + b * cos( (PI*2.0) * (c * t + d));
}

mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

float torus(vec3 p,vec2 t) {

    vec2 q = vec2(length(vec2(p.x,p.z)) - t.x,p.y);
    return length(q) - t.y; 
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

p += 0.025 * sin3(p+noise(p),10.  );

res = vec2(torus(p,vec2(2.,1.5)) ,1.) ;

return res;
}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float depth = 0.0;
    float d = -1.0;

    for(int i = 0; i < 1500; i++) {

        vec3 p = ro + depth * rd;
        vec2 dist = scene(p);
   
        if(abs( dist.x) < 0.001 || 500. <  dist.x ) { break; }
        depth += dist.x;
        d = dist.y;

        }
 
        if(500. < depth) { d = -1.0; }
        return vec2(depth,d);

}


float shadow(vec3 ro,vec3 rd,float dmin,float dmax) {

    float res = 1.0;
    float t = dmin;
    float ph = 1e10;
    
    for(int i = 0; i < 6; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float s = clamp(8.0*h/t,0.0,1.0);
        res = min(res,s*s*(3.-2. *s ));         
        t += clamp(h,0.02,0.1 );
    
        if(res < 0.001 || t > dmax ) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * 0.001;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));

}

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));


     vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {

float t = u_time;

vec3 col = vec3(0.);

vec2 d = rayScene(ro, rd);

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize( vec3(0. ));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

p.xz *= rot2(t * 0.0001);

col =  vec3(0.) * d.y;

float fres = 0.;
float ns = 0.;

if(d.y >= 1.) {
fres = 10.;

        ns += fractal(p+fractal(p+fractal(p,5,.25),5,.5),6,.5); 
        
         
        col = fmCol(p.y+ns,vec3(.25),
                           vec3(.25), 
                           vec3(.15),
                           vec3(.25));
    
} else {

    fres = 1.;
    col = vec3(1.0);

}


float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);
float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * (.04 + 0.75 * pow(clamp(1. + dot(h,rd),0.,1.),5.));
float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

dif *= shadow(p,l,0.02,5.);

vec3 linear = vec3(0.);
linear += 1. * dif  * vec3(.5);
linear += fres * fre * vec3(1.);

col = col * linear;
   
col = pow(col,vec3(.4545));

return col;
}

void main() {
 
vec3 cam_target = vec3(0.0);
vec3 cam_pos = vec3(.25,1.25,.75);
cam_pos = cameraPosition;

vec2 uvu = -1.0 + 2.0 * uVu.xy;
uvu.x *= u_resolution.x/u_resolution.y; 

vec3 direction = rayCamDir(uvu,cam_pos,cam_target,1.);
vec3 color = render(cam_pos,direction);

out_FragColor = vec4(color,1.0);

}
