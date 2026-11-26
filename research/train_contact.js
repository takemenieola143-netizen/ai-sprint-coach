/* Run: node research/train_contact.js dataset.json model.json evaluation.json
Dataset: JSON {clips:[{id,athlete,split,license,source,groundTruth,timeVerified,width,height,playbackTimeScale,rows,events,completeAnnotation}]}.
Only use lawfully reusable data. No raw videos needed. Event timestamps are REAL seconds, from the clip's time origin.
*/
'use strict';
const fs=require('node:fs');
const A=require('./contact_model.js');
function run(dataset){
 const clips=dataset.clips;if(!Array.isArray(clips)||!clips.length)throw Error('No clips');
 const athletes=new Map(),ids=new Set(),sources=new Set(),groups={train:[],validation:[],test:[]};
 for(const c of clips){
  if(!['train','validation','test'].includes(c.split)||typeof c.athlete!=='string'||!c.athlete||typeof c.id!=='string'||!c.id||ids.has(c.id)||typeof c.source!=='string'||!c.source||sources.has(c.source))throw Error('Require unique clip ID/source and athlete split');
  if(athletes.has(c.athlete)&&athletes.get(c.athlete)!==c.split)throw Error('Athlete leakage: '+c.athlete);
  athletes.set(c.athlete,c.split);ids.add(c.id);sources.add(c.source);
  if(!c.license||c.license==='unknown'||c.license==='pending'||!['force','independent-video'].includes(c.groundTruth)||c.timeVerified!==true||c.completeAnnotation!==true)throw Error('Unverified licensing, timing or incomplete independent labels: '+c.id);
  if(!Array.isArray(c.rows)||!c.rows.length||!Array.isArray(c.events)||!Number.isFinite(c.width)||c.width<=0||!Number.isFinite(c.height)||c.height<=0||!Number.isFinite(c.playbackTimeScale)||c.playbackTimeScale<=0)throw Error('Invalid clip '+c.id);
  const unique=A.prepare(c.rows,c.playbackTimeScale);if(!unique.length)throw Error('No distinct frames: '+c.id);
  const keys=new Set();for(let e of c.events){let k=e.side+'|'+e.type+'|'+e.t;if(!['Left','Right'].includes(e.side)||!['touchdown','toeoff'].includes(e.type)||!Number.isFinite(e.t)||e.t<unique[0].t-.025||e.t>unique.at(-1).t+.025||keys.has(k))throw Error('Invalid or duplicate event '+c.id);keys.add(k)}
  groups[c.split].push({...c,features:A.extract(c.rows,c.width,c.height,c.playbackTimeScale)});
 }
 if(Object.values(groups).some(g=>!g.length))throw Error('All three athlete-disjoint splits are required');
 let samples=[];
 for(let c of groups.train)for(let w of c.features.windows){
  let candidates=c.events.filter(e=>e.side===w.side).map(e=>({...e,d:Math.abs(e.t-w.t)})).sort((a,b)=>a.d-b.d);
  let near=candidates[0],y=0;if(near&&near.d<=.025)y=A.CLASSES.indexOf(near.type);else if(near&&near.d<.05)continue;
  samples.push({x:w.x,y});
 }
 let model=A.fit(samples);
 function score(group,threshold){
  let perClip=group.map(c=>({id:c.id,athlete:c.athlete,groundTruth:c.groundTruth,windows:c.features.windows.length,possibleWindows:c.features.possibleWindows,...A.evaluate(A.events(model,c.features.windows,threshold),c.events)}));
  let sum=k=>perClip.reduce((s,x)=>s+x[k],0),tp=sum('tp'),fp=sum('fp'),fn=sum('fn');
  return {threshold,tp,fp,fn,precision:tp+fp?tp/(tp+fp):null,recall:tp+fn?tp/(tp+fn):null,f1:2*tp+fp+fn?2*tp/(2*tp+fp+fn):null,maeMs:tp?perClip.reduce((s,x)=>s+x.tp*(x.maeMs??0),0)/tp:null,rmseMs:tp?Math.sqrt(perClip.reduce((s,x)=>s+x.tp*(x.rmseMs??0)**2,0)/tp):null,windowCoverage:sum('possibleWindows')?sum('windows')/sum('possibleWindows'):null,perClip};
 }
 let tuning=[.50,.60,.65,.70,.80,.90].map(t=>score(groups.validation,t)).sort((a,b)=>(b.f1??0)-(a.f1??0)||b.threshold-a.threshold);
 model.threshold=tuning[0].threshold;
 const counts=Object.fromEntries(Object.entries(groups).map(([k,g])=>[k,{clips:g.length,athletes:new Set(g.map(c=>c.athlete)).size,truthEvents:g.reduce((s,c)=>s+c.events.length,0)}]));
 // Validation scores do not authorize release. Compare current baseline on same held-out outdoor sprint clips first.
 return {model,evaluation:{featureVersion:A.VERSION,counts,trainingWindows:samples.length,toleranceMs:20,validation:tuning[0],test:score(groups.test,model.threshold),releaseApproved:false,limitations:'No automatic release. Audit clip duplicates, license, source timing, outdoor domain, baseline and event/phase annotation quality independently.'}};
}
if(require.main===module){const [input,modelPath,evalPath]=process.argv.slice(2);if(!input||!modelPath||!evalPath)throw Error('Usage: node research/train_contact.js dataset.json model.json evaluation.json');let r=run(JSON.parse(fs.readFileSync(input,'utf8')));fs.writeFileSync(modelPath,JSON.stringify(r.model));fs.writeFileSync(evalPath,JSON.stringify(r.evaluation,null,2));}
module.exports={run};
