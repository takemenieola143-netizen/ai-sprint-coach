/* AI Sprint Coach: temporal contact feature extractor and lightweight model.
 * No external dependencies. Features are 2D projected landmarks, not ground truth.
 * Runtime: browser global SprintContactResearch or Node CommonJS.
 */
(function(root){
'use strict';
const OFFSETS=[-.100,-.050,0,.050,.100], JOINTS=[23,25,27,29,31], DIM=OFFSETS.length*12;
const VERSION='contact-window-v1', CLASSES=['none','touchdown','toeoff'];
function visible(p){return p&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1&&p.visibility>=.7}
function prepare(rows,scale=1){
 if(!Number.isFinite(scale)||scale<=0)throw Error('Invalid time scale');
 let result=[],last=-Infinity;
 for(let i=0;i<rows.length;i++){let r=rows[i],t=r.seekSeconds*scale;if(r.duplicate||!Number.isFinite(t)||t<=last)continue;result.push({index:i,t,p:r.points||[]});last=t}
 return result;
}
function nearest(rows,t){let lo=0,hi=rows.length;while(lo<hi){let m=(lo+hi)>>1;if(rows[m].t<t)lo=m+1;else hi=m}let a=rows[lo],b=rows[lo-1];return !a?b:!b?a:Math.abs(a.t-t)<Math.abs(b.t-t)?a:b}
function windowFeatures(rows,j,side,W,H){
 let ref=rows[j];if(!ref||!(W>0&&H>0)||!(side===0||side===1))return null;
 let p=ref.p,hip=p[23+side],knee=p[25+side],ank=p[27+side];
 if(![hip,knee,ank,p[31+side]].every(visible))return null;
 let dist=(a,b)=>Math.hypot((a.x-b.x)*W,(a.y-b.y)*H),L=dist(hip,knee)+dist(knee,ank);
 if(L<25)return null;
 let x=[],used=new Set(),goodFrames=0;
 for(let dt of OFFSETS){
  let r=nearest(rows,ref.t+dt);if(!r||Math.abs(r.t-ref.t-dt)>.025||used.has(r.index))return null;
  used.add(r.index);let h=r.p[23+side];if(!visible(h))return null;
  let validFoot=visible(r.p[31+side])&&visible(r.p[27+side]);if(validFoot)goodFrames++;
  for(let k of JOINTS.slice(1)){let q=r.p[k+side];if(visible(q)){x.push((q.x-h.x)*W/L,(q.y-h.y)*H/L,1)}else{x.push(0,0,0)}}
 }
 if(goodFrames<4)return null;
 return {x,index:ref.index,t:ref.t,side:side?'Right':'Left'};
}
function extract(rows,W,H,scale=1){
 let p=prepare(rows,scale),out=[];
 for(let j=0;j<p.length;j++)for(let s=0;s<2;s++){let f=windowFeatures(p,j,s,W,H);if(f)out.push(f)}
 return {featureVersion:VERSION,dim:DIM,distinctSamples:p.length,possibleWindows:p.length*2,windows:out};
}
function softmax(z){let m=Math.max(...z),e=z.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s)}
function predict(model,x){if(model.featureVersion!==VERSION||!Array.isArray(x)||x.length!==DIM)throw Error('Feature mismatch');return softmax(model.weights.map((w,k)=>model.bias[k]+w.reduce((s,a,i)=>s+a*x[i],0)))}
function fit(samples,opts={}){
 if(!samples.length)throw Error('No training windows');
 const counts=[0,0,0];for(let s of samples){if(!Number.isInteger(s.y)||s.y<0||s.y>2||s.x.length!==DIM||!s.x.every(Number.isFinite))throw Error('Invalid sample');counts[s.y]++}
 if(counts.some(x=>x===0))throw Error('All three event classes required');
 const weight=counts.map(n=>Math.min(20,samples.length/(3*n)));
 let model={featureVersion:VERSION,classes:CLASSES,weights:Array.from({length:3},()=>Array(DIM).fill(0)),bias:[0,0,0],validated:false};
 let epochs=opts.epochs??100,lr=opts.lr??.15,l2=.001;
 for(let e=0;e<epochs;e++){
  let g=Array.from({length:3},()=>Array(DIM).fill(0)),b=[0,0,0],den=0;
  for(let s of samples){let p=predict(model,s.x),w=weight[s.y];den+=w;for(let k=0;k<3;k++){let d=w*(p[k]-(k===s.y?1:0));b[k]+=d;for(let i=0;i<DIM;i++)g[k][i]+=d*s.x[i]}}
  for(let k=0;k<3;k++){model.bias[k]-=lr*b[k]/den;for(let i=0;i<DIM;i++)model.weights[k][i]-=lr*(g[k][i]/den+l2*model.weights[k][i])}
 }
 return model;
}
function events(model,windows,threshold=.65,refractory=.10){
 let c=[];for(let w of windows){let p=predict(model,w.x),k=p[1]>p[2]?1:2;if(p[k]>=threshold&&p[k]>p[0])c.push({side:w.side,type:CLASSES[k],t:w.t,index:w.index,score:p[k]})}
 c.sort((a,b)=>b.score-a.score);let kept=[];for(let x of c)if(!kept.some(y=>y.side===x.side&&y.type===x.type&&Math.abs(y.t-x.t)<refractory))kept.push(x);
 return kept.sort((a,b)=>a.t-b.t)
}
function evaluate(pred,truth,tolerance=.020){
 let pairs=[];for(let i=0;i<pred.length;i++)for(let j=0;j<truth.length;j++)if(pred[i].side===truth[j].side&&pred[i].type===truth[j].type&&Math.abs(pred[i].t-truth[j].t)<=tolerance)pairs.push({i,j,d:pred[i].t-truth[j].t});
 pairs.sort((a,b)=>Math.abs(a.d)-Math.abs(b.d));let P=new Set(),T=new Set(),err=[];for(let q of pairs)if(!P.has(q.i)&&!T.has(q.j)){P.add(q.i);T.add(q.j);err.push(q.d)}
 let tp=err.length,fp=pred.length-tp,fn=truth.length-tp;
 return {tp,fp,fn,precision:pred.length?tp/pred.length:null,recall:truth.length?tp/truth.length:null,maeMs:tp?1000*err.reduce((s,x)=>s+Math.abs(x),0)/tp:null,rmseMs:tp?1000*Math.sqrt(err.reduce((s,x)=>s+x*x,0)/tp):null};
}
const api={VERSION,CLASSES,DIM,prepare,extract,fit,predict,events,evaluate};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.SprintContactResearch=api;
})(typeof globalThis==='undefined'?this:globalThis);
