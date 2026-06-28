#include <metal_stdlib>
using namespace metal;

struct RenderParams { uint width; uint height; uint nr; uint ntheta; uint nphi; float spin_a; uint max_steps; float step; };
struct State6 { float t; float r; float th; float ph; float pr; float pth; };

inline float wrap_phi(float x) { float y = fmod(x, 6.28318530718f); return y < 0.0f ? y + 6.28318530718f : y; }
inline uint cidx(uint ir, uint it, uint ip, uint c, uint nt, uint np) { return (((ir * nt) + it) * np + ip) * 11u + c; }

float4x4 metric_contravariant(float r, float theta, float a) {
    float ct = cos(theta), st = sin(theta); float s2 = max(st * st, 1.0e-8f);
    float sig = r * r + a * a * ct * ct; float dlt = r * r - 2.0f * r + a * a;
    float A = (r * r + a * a) * (r * r + a * a) - a * a * dlt * s2;
    float4x4 g = float4x4(0.0f);
    g[0][0] = -A / (sig * dlt); g[0][3] = -2.0f * a * r / (sig * dlt); g[3][0] = g[0][3];
    g[1][1] = dlt / sig; g[2][2] = 1.0f / sig; g[3][3] = (dlt - a * a * s2) / (sig * dlt * s2);
    return g;
}

void metric_derivative_numeric(float r, float th, float a, thread float4x4& dr, thread float4x4& dth) {
    float er = 1.0e-3f, et = 1.0e-4f;
    float4x4 gr0 = metric_contravariant(r-er, th, a), gr1 = metric_contravariant(r+er, th, a);
    float4x4 gt0 = metric_contravariant(r, th-et, a), gt1 = metric_contravariant(r, th+et, a);
    dr = (gr1 - gr0) / (2.0f * er); dth = (gt1 - gt0) / (2.0f * et);
}

State6 kerr_rhs(State6 y, float p_t, float p_phi, float a) {
    float4x4 g = metric_contravariant(y.r, y.th, a); float4x4 dr, dth; metric_derivative_numeric(y.r, y.th, a, dr, dth);
    float p[4] = {p_t, y.pr, y.pth, p_phi}; float xd[4] = {0,0,0,0};
    for (uint mu=0; mu<4; ++mu) for (uint nu=0; nu<4; ++nu) xd[mu] += g[mu][nu] * p[nu];
    float pr = 0.0f, pth = 0.0f;
    for (uint mu=0; mu<4; ++mu) for (uint nu=0; nu<4; ++nu) { pr += -0.5f * p[mu] * dr[mu][nu] * p[nu]; pth += -0.5f * p[mu] * dth[mu][nu] * p[nu]; }
    return State6{xd[0], xd[1], xd[2], xd[3], pr, pth};
}

State6 rk2_geodesic_step(State6 y, float h, float p_t, float p_phi, float a) {
    State6 k1 = kerr_rhs(y, p_t, p_phi, a);
    State6 m = State6{y.t+.5f*h*k1.t, y.r+.5f*h*k1.r, y.th+.5f*h*k1.th, y.ph+.5f*h*k1.ph, y.pr+.5f*h*k1.pr, y.pth+.5f*h*k1.pth};
    State6 k2 = kerr_rhs(m, p_t, p_phi, a);
    return State6{y.t+h*k2.t, y.r+h*k2.r, clamp(y.th+h*k2.th, 1.0e-6f, 3.1415926f-1.0e-6f), y.ph+h*k2.ph, y.pr+h*k2.pr, y.pth+h*k2.pth};
}

bool bracket_linear(device const float* grid, uint n, float x, thread uint& i0, thread uint& i1, thread float& w) {
    if (x < grid[0] || x > grid[n-1]) return false; uint lo = 0; while (lo < n-2 && grid[lo+1] <= x) ++lo; i0 = lo; i1 = lo + 1; w = (x - grid[i0]) / max(grid[i1] - grid[i0], 1.0e-20f); return true;
}
void bracket_phi(device const float* grid, uint n, float ph, thread uint& i0, thread uint& i1, thread float& w) {
    float p = wrap_phi(ph); uint lo = 0; while (lo < n-1 && grid[lo+1] <= p) ++lo; i0 = lo; i1 = (lo + 1) % n; float hi = i1 > i0 ? grid[i1] : grid[0] + 6.28318530718f; float pp = p >= grid[i0] ? p : p + 6.28318530718f; w = (pp - grid[i0]) / max(hi - grid[i0], 1.0e-20f);
}

void sample_brick_trilinear(device const float* coeffs, device const float* rg, device const float* tg, device const float* pg, uint nr, uint nt, uint np, float r, float th, float ph, thread float outv[11]) {
    uint r0,r1,t0,t1,p0,p1; float wr,wt,wp;
    if (!bracket_linear(rg,nr,r,r0,r1,wr) || !bracket_linear(tg,nt,th,t0,t1,wt)) { for (uint c=0;c<11;++c) outv[c] = 0.0f; return; }
    bracket_phi(pg,np,ph,p0,p1,wp);
    for (uint c=0;c<11;++c) {
        float c000=coeffs[cidx(r0,t0,p0,c,nt,np)], c001=coeffs[cidx(r0,t0,p1,c,nt,np)], c010=coeffs[cidx(r0,t1,p0,c,nt,np)], c011=coeffs[cidx(r0,t1,p1,c,nt,np)];
        float c100=coeffs[cidx(r1,t0,p0,c,nt,np)], c101=coeffs[cidx(r1,t0,p1,c,nt,np)], c110=coeffs[cidx(r1,t1,p0,c,nt,np)], c111=coeffs[cidx(r1,t1,p1,c,nt,np)];
        float c00=mix(c000,c001,wp), c01=mix(c010,c011,wp), c10=mix(c100,c101,wp), c11=mix(c110,c111,wp); outv[c]=mix(mix(c00,c01,wt), mix(c10,c11,wt), wr);
    }
}

void stokes_rhs(thread const float S[4], thread const float c[11], thread float o[4]) { o[0]=c[0]-(c[4]*S[0]+c[5]*S[1]+c[6]*S[2]+c[7]*S[3]); o[1]=c[1]-(c[5]*S[0]+c[4]*S[1]+c[8]*S[2]-c[9]*S[3]); o[2]=c[2]-(c[6]*S[0]-c[8]*S[1]+c[4]*S[2]+c[10]*S[3]); o[3]=c[3]-(c[7]*S[0]+c[9]*S[1]-c[10]*S[2]+c[4]*S[3]); }
void stokes_step_rk2(thread float S[4], thread const float c[11], float ds) { float k1[4],m[4],k2[4]; stokes_rhs(S,c,k1); for(uint i=0;i<4;++i)m[i]=S[i]+.5f*ds*k1[i]; stokes_rhs(m,c,k2); for(uint i=0;i<4;++i)S[i]+=ds*k2[i]; }

kernel void kerr_stokes_render_kernel(device float4* out_stokes [[buffer(0)]], device const float* coeffs [[buffer(1)]], device const float* r_grid [[buffer(2)]], device const float* theta_grid [[buffer(3)]], device const float* phi_grid [[buffer(4)]], constant RenderParams& params [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.width || gid.y >= params.height) return; uint pix = gid.y * params.width + gid.x;
    float ndcx = 2.0f * ((float(gid.x) + .5f) / float(params.width)) - 1.0f; float ndcy = 1.0f - 2.0f * ((float(gid.y) + .5f) / float(params.height));
    State6 s = State6{0.0f,55.0f,1.134464f,0.0f,-1.0f,-.22f*ndcy}; float p_t=-1.0f,p_phi=.12f*ndcx,S[4]={0,0,0,0}; float rplus=1.0f+sqrt(max(1.0f-params.spin_a*params.spin_a,0.0f));
    for(uint n=0;n<params.max_steps;++n){ State6 prev=s; s=rk2_geodesic_step(s,params.step,p_t,p_phi,params.spin_a); float c[11]; sample_brick_trilinear(coeffs,r_grid,theta_grid,phi_grid,params.nr,params.ntheta,params.nphi,.5f*(prev.r+s.r),.5f*(prev.th+s.th),.5f*(prev.ph+s.ph),c); float ds=length(float3(s.r-prev.r,s.th-prev.th,s.ph-prev.ph)); stokes_step_rk2(S,c,ds); if(s.r<=rplus*1.0002f||s.r>220.0f)break; }
    out_stokes[pix] = float4(S[0],S[1],S[2],S[3]);
}
