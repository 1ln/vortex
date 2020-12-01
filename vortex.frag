#version 300 es     

//dolson
//2020

out vec4 FragColor;
varying vec2 uVu;

uniform vec2 resolution;
uniform vec4 mouse;
uniform float time;
uniform int seed; 

uniform sampler2D tex;

mat2 m = mat2(0.8,0.6,-0.6,0.8);

float h(float p) {

p = fract(p*0.1493);
p += p * (float(seed) * 0.000001); 
return p;

}

float n(vec2 p) {
return sin(p.x)*sin(p.y);
}

float f6(vec2 p) {

    float f = 1.;

    f += 0.5      * n(p); p = m*p*2.01;
    f += 0.25     * n(p); p = m*p*2.02;
    f += 0.125    * n(p); p = m*p*2.04;
    f += 0.0625   * n(p); p = m*p*2.03;
    f += 0.0325   * n(p); p = m*p*2.06;
    f += 0.015625 * n(p);
    return f/0.92;    

}

float dd(vec2 p) {

vec2 q = vec2(f6(p + vec2(0.0,1.0)),
              f6(p + vec2(h(103.)*25.,1.5)));

vec2 r = vec2(f6(p + 4.0 * q + vec2(h(144.)*45.,4.8)),
              f6(p + 4.0 * q + vec2(6.8,9.1)));

return f6(p + 4.0 * r);
}

float hyperbola(vec3 p) { 

vec2 l = vec2(length(p.xz) ,-p.y);
float a = 0.5;
float d = sqrt((l.x+l.y)*(l.x+l.y)- 4. *(l.x*l.y-a)) + 0.5; 
return (-l.x-l.y+d)/2.0;

}

void main() {
 
    vec3 col;

    vec2 uv = (2. * gl_FragCoord.xy - 
    resolution.xy) / resolution.y; 
    
    float ra = time*0.00001;
    mat2 r = mat2(cos(ra),sin(ra),-sin(ra),cos(ra));

    float fov = 1.0;
    float a = -0.75;
    
    vec3 p = vec3(0.0,-0.75,-1.0);

    vec3 d = vec3(uv*fov,1.);
    d.yz *= mat2(cos(a),sin(a),-sin(a),cos(a));

    for(int i = 0; i < 100; ++i) {
        p += d * hyperbola(p);        
    }

    float fd = 3.0;

    float fs = 2.;
    float fa = 0.125;
    
    float nl = dd(p.xz*r);
    nl += f6(p.xz+f6(p.yx)) * dot(uv,uv);
    nl += mix(dot(mouse.xy,p.xz),dot(nl*p.xy,mouse.xy),fa*0.001);

    col = vec3(nl*log((p.y+fd)*fs)/log((fd-fa)*fs));
    col += 0.005*texture(tex,uVu).xyz;
    
    col = pow(col,vec3(0.4545));
    FragColor = vec4(col,1.);

}
