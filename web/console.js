const R=window.RESULTS,M=R.meta,T=R.tracks,L=M.leads,[la0,la1,lo0,lo1]=M.domain,S=640,$=id=>document.getElementById(id);
const X=o=>(o-lo0)/(lo1-lo0)*S,Y=a=>(la1-a)/(la1-la0)*S,kmpx=k=>k/111/(la1-la0)*S,ok=p=>p&&p[0]!=null;
const st={i:8,tab:"h"};const lead=$("lead");lead.max=L.length-1;lead.value=st.i;
const kv=(a)=>a.map(r=>`<span>${r[0]}</span><span>${r[1]}</span>`).join("");
$("ev").innerHTML=kv([["Case","Synthetic cyclone"],["Region","Bay of Bengal"],["Members",M.members],["Provenance",M.provenance],["Peak EFI",R.efi.peak]]);
const pct=(a,q)=>{a=a.filter(x=>x!=null).sort((x,y)=>x-y);return a[Math.floor(q*(a.length-1))]};
function map(){const i=st.i;let s=`<rect width="${S}" height="${S}" fill="#f4f7f9"/><path d="${window.LAND}" fill="#e6e8e0" stroke="#8b9188" stroke-width=".8"/>`;
for(let la=Math.ceil(la0/2)*2;la<=la1;la+=2)s+=`<line x1="0" x2="${S}" y1="${Y(la)}" y2="${Y(la)}" stroke="#e1e3dc"/><text x="4" y="${Y(la)-3}" font-size="11" fill="#5b6470">${la}°N</text>`;
for(let lo=Math.ceil(lo0/2)*2;lo<=lo1;lo+=2)s+=`<line y1="0" y2="${S}" x1="${X(lo)}" x2="${X(lo)}" stroke="#e1e3dc"/><text y="${S-4}" x="${X(lo)+3}" font-size="11" fill="#5b6470">${lo}°E</text>`;
if($("lc").checked)for(let k=0;k<=i;k++)if(ok(T.mean[k])&&T.r90[k]!=null)s+=`<circle cx="${X(T.mean[k][1])}" cy="${Y(T.mean[k][0])}" r="${kmpx(T.r90[k])}" fill="${k==i?"#1f4e79":"none"}" fill-opacity=".10" stroke="#1f4e79" stroke-opacity="${k==i?.7:.18}"/>`;
if($("lm").checked)T.members.forEach(m=>{const p=m.slice(0,i+1).filter(ok).map(q=>X(q[1])+","+Y(q[0])).join(" ");s+=`<polyline points="${p}" fill="none" stroke="#6b7c93" stroke-opacity=".35" stroke-width="1"/>`;
 if(ok(m[i]))s+=`<circle cx="${X(m[i][1])}" cy="${Y(m[i][0])}" r="2.6" fill="#33445c"/>`});
s+=`<polyline points="${T.mean.slice(0,i+1).filter(ok).map(q=>X(q[1])+","+Y(q[0])).join(" ")}" fill="none" stroke="#1f4e79" stroke-width="2.2"/>`;
if($("lt").checked)s+=`<polyline points="${T.truth.slice(0,i+1).map(q=>X(q[1])+","+Y(q[0])).join(" ")}" fill="none" stroke="#b3261e" stroke-width="1.6" stroke-dasharray="5 4"/>`;
if($("lb").checked){const c=T.mean[M.crop_lead_idx],w=kmpx(M.dx_km*64);s+=`<rect x="${X(c[1])-w/2}" y="${Y(c[0])-w/2}" width="${w}" height="${w}" fill="none" stroke="#d9730d" stroke-width="1.6"/><text x="${X(c[1])+w/2+4}" y="${Y(c[0])-w/2+12}" font-size="11" fill="#d9730d">5 km crop, +${L[M.crop_lead_idx]} h</text>`}
s+=`<circle cx="${X(T.target[1])}" cy="${Y(T.target[0])}" r="${kmpx(T.strike_R_km)}" fill="none" stroke="#333" stroke-dasharray="2 3"/><text x="${X(T.target[1])+8}" y="${Y(T.target[0])+kmpx(T.strike_R_km)+14}" font-size="11" fill="#333">reference point, 150 km</text>`;
$("map").innerHTML=s;$("lv").textContent="+"+L[i]+" h";
const v=T.vmax.map(m=>m[i]).filter(x=>x!=null);
$("ens").innerHTML=kv([["Members tracked",T.members.filter(m=>ok(m[i])).length+" / "+M.members],["Spread (90%)",T.r90[i]==null?"-":T.r90[i]+" km"],["Ensemble-mean centre",ok(T.mean[i])?T.mean[i][0].toFixed(2)+"N "+T.mean[i][1].toFixed(2)+"E":"-"],["Max wind, median",Math.round(pct(v,.5))+" km/h"],["Max wind, 10–90%",Math.round(pct(v,.1))+"–"+Math.round(pct(v,.9))],["P(within 150 km of reference)",(T.strike[i]*100).toFixed(0)+"%"]])}
function heat(id,a,lo,hi,c0,c1,cap){const n=a.length,cv=$(id);cv.width=n;cv.height=n;const g=cv.getContext("2d"),im=g.createImageData(n,n);
 a.forEach((row,y)=>row.forEach((val,x)=>{const f=Math.min(1,Math.max(0,(val-lo)/(hi-lo))),o=(y*n+x)*4;for(let k=0;k<3;k++)im.data[o+k]=c0[k]+(c1[k]-c0[k])*f;im.data[o+3]=255}));g.putImageData(im,0,0)}
const fig=(id,t)=>`<figure><canvas id="${id}"></canvas><figcaption>${t}</figcaption></figure>`;
function line(w,h,series,xl,yl){const all=series.flatMap(s=>s.d),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),p=32,
 sx=x=>p+(x-x0)/(x1-x0)*(w-p-8),sy=y=>h-p-(y-y0)/(y1-y0||1)*(h-p-8);
 return `<svg viewBox="0 0 ${w} ${h}" width="100%"><rect x="${p}" y="8" width="${w-p-8}" height="${h-p-8}" fill="#fff" stroke="#c9ccc4"/>`+series.map(s=>`<polyline fill="none" stroke="${s.c}" stroke-width="1.6" ${s.dash?'stroke-dasharray="4 3"':""} points="${s.d.map(q=>sx(q[0])+","+sy(q[1])).join(" ")}"/>`).join("")+
 `<text x="${w/2}" y="${h-6}" font-size="11" text-anchor="middle" fill="#5b6470">${xl}</text><text x="10" y="${h/2}" font-size="11" fill="#5b6470" transform="rotate(-90 10 ${h/2})" text-anchor="middle">${yl}</text></svg>`}
function pane(){const p=$("pane"),D=R.downscale,me=R.metrics;
if(st.tab=="h"){p.innerHTML=`<div class="cv">${fig("c1","Coarse input (10 km)")}${fig("c2","Bicubic interpolation")}${fig("c3","Mean of 32 samples")}${fig("c4","Reference (synthetic truth)")}${fig("c5","P(wind &gt; 88 km/h)")}${fig("c6","P(wind &gt; 118 km/h)")}</div><p class="note">Wind panels share a 0–220 km/h scale. Probability panels come from ${R.alert.n_samples} samples. The sample mean is smoother than any single sample; alerts use the samples, not the mean.</p>`;
 const b=[241,244,248],e=[31,78,121],r0=[255,247,236],r1=[179,0,0];heat("c1",D.coarse,0,220,b,e);heat("c2",D.bicubic,0,220,b,e);heat("c3",D.sample_mean,0,220,b,e);heat("c4",D.truth,0,220,b,e);heat("c5",D.p_mod,0,100,r0,r1);heat("c6",D.p_sev,0,100,r0,r1)}
else if(st.tab=="e"){const t=me.track,d=me.downscale,ix=t.lead.map((_,k)=>k).filter(k=>k%4==0&&t.ens_mean_err[k]!=null);
 p.innerHTML=`<h3>Track error (${t.n_cases} synthetic cases)</h3><table><tr><th>Lead</th><th class="n">Ens. mean</th><th class="n">Control</th><th class="n">Spread</th><th class="n">SSR</th></tr>${ix.map(k=>`<tr><td>+${t.lead[k]} h</td><td class="n">${t.ens_mean_err[k]}</td><td class="n">${t.control_err[k]??"-"}</td><td class="n">${t.spread[k]}</td><td class="n">${t.ssr[k]??"-"}</td></tr>`).join("")}</table><p class="note">km. SSR above 1: the synthetic ensemble is over-dispersed.</p>
 <h3>Strike-probability reliability, +96 h</h3>${line(300,230,[{c:"#aaa",dash:1,d:[[0,0],[1,1]]},{c:"#1f4e79",d:me.reliability.map(r=>[r.p,r.obs])}],"forecast probability","observed frequency")}
 <h3>Downscaling (20 held-out fields)</h3><table><tr><th></th><th class="n">RMSE</th><th class="n">Peak ratio</th><th class="n">Top 0.1% err</th></tr>${["bicubic","regression","stochastic_sample"].map(k=>`<tr><td>${k.replace("_"," ")}</td><td class="n">${d[k].rmse.toFixed(2)}</td><td class="n">${d[k].peak_ratio.toFixed(3)}</td><td class="n">${d[k].q999_err_pct.toFixed(2)}%</td></tr>`).join("")}</table>
 <p class="note">CRPS: ${d.crps.ensemble16.toFixed(2)} (16 samples) vs ${d.crps.bicubic.toFixed(2)} (bicubic). A single sample has higher RMSE by design (it is one plausible realisation, not the mean).</p>
 <h3>Power spectrum</h3>${line(300,230,["truth:#111","bicubic:#b3261e","regression:#d9730d","sample:#1f4e79"].map(s=>{const[n,c]=s.split(":");return{c,d:me.psd.k.map((k,j)=>[Math.log10(k),me.psd[n][j]])}}),"log10 wavenumber","log10 power")}
 <p class="note">Black = truth, red = bicubic, orange = regression mean, blue = one sample. Bicubic and regression lose power at small scales; the diffusion sample slightly overshoots them. Toy 2x task on synthetic data.</p>`}
else{const a=R.alert;p.innerHTML=`<div class="kv"><span>Category</span><span><span class="cat ${a.colour}">${a.category}</span></span><span>Valid for lead</span><span>+${a.lead_h} h</span><span>Core coordinate</span><span>${a.core[0].toFixed(2)}N ${a.core[1].toFixed(2)}E</span><span>P(wind &gt; ${a.threshold_kmh} km/h) at core</span><span>${(a.core_prob*100).toFixed(0)}%</span><span>Samples</span><span>${a.n_samples}</span></div><p class="note">${a.guidance} Thresholds are illustrative.</p>
 <details open><summary>Alert JSON</summary><pre>${JSON.stringify(a,null,1)}</pre></details><details><summary>CAP 1.2</summary><pre></pre></details><details><summary>GeoJSON</summary><pre></pre></details>`;
 const pr=p.querySelectorAll("pre");pr[1].textContent=R.cap;pr[2].textContent=JSON.stringify(R.geojson,null,1)}}
lead.oninput=()=>{st.i=+lead.value;map()};["lm","lc","lt","lb"].forEach(k=>$(k).onchange=map);
document.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".tabs button").forEach(x=>x.classList.remove("on"));b.classList.add("on");st.tab=b.dataset.t;pane()});
map();pane();
