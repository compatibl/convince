"use strict";(self.webpackChunkui_react=self.webpackChunkui_react||[]).push([[1834],{74741:(t,e,n)=>{n.d(e,{Wi:()=>h,Z0:()=>l,aU:()=>o,eZ:()=>c,wY:()=>a,xw:()=>u});var s=n(4669),i=n(5976),r=n(63580);class o extends i.JT{constructor(t,e="",n="",i=!0,r){super(),this._onDidChange=this._register(new s.Q5),this.onDidChange=this._onDidChange.event,this._enabled=!0,this._id=t,this._label=e,this._cssClass=n,this._enabled=i,this._actionCallback=r}get id(){return this._id}get label(){return this._label}set label(t){this._setLabel(t)}_setLabel(t){this._label!==t&&(this._label=t,this._onDidChange.fire({label:t}))}get tooltip(){return this._tooltip||""}set tooltip(t){this._setTooltip(t)}_setTooltip(t){this._tooltip!==t&&(this._tooltip=t,this._onDidChange.fire({tooltip:t}))}get class(){return this._cssClass}set class(t){this._setClass(t)}_setClass(t){this._cssClass!==t&&(this._cssClass=t,this._onDidChange.fire({class:t}))}get enabled(){return this._enabled}set enabled(t){this._setEnabled(t)}_setEnabled(t){this._enabled!==t&&(this._enabled=t,this._onDidChange.fire({enabled:t}))}get checked(){return this._checked}set checked(t){this._setChecked(t)}_setChecked(t){this._checked!==t&&(this._checked=t,this._onDidChange.fire({checked:t}))}async run(t,e){this._actionCallback&&await this._actionCallback(t)}}class h extends i.JT{constructor(){super(...arguments),this._onWillRun=this._register(new s.Q5),this.onWillRun=this._onWillRun.event,this._onDidRun=this._register(new s.Q5),this.onDidRun=this._onDidRun.event}async run(t,e){if(!t.enabled)return;let n;this._onWillRun.fire({action:t});try{await this.runAction(t,e)}catch(t){n=t}this._onDidRun.fire({action:t,error:n})}async runAction(t,e){await t.run(e)}}class l{constructor(){this.id=l.ID,this.label="",this.tooltip="",this.class="separator",this.enabled=!1,this.checked=!1}static join(...t){let e=[];for(const n of t)n.length&&(e=e.length?[...e,new l,...n]:n);return e}async run(){}}l.ID="vs.actions.separator";class a{get actions(){return this._actions}constructor(t,e,n,s){this.tooltip="",this.enabled=!0,this.checked=void 0,this.id=t,this.label=e,this.class=s,this._actions=n}async run(){}}class c extends o{constructor(){super(c.ID,r.NC("submenu.empty","(empty)"),void 0,!1)}}function u(t){var e,n;return{id:t.id,label:t.label,tooltip:null!==(e=t.tooltip)&&void 0!==e?e:t.label,class:t.class,enabled:null===(n=t.enabled)||void 0===n||n,checked:t.checked,run:async(...e)=>t.run(...e)}}c.ID="vs.actions.empty"},9488:(t,e,n)=>{function s(t,e=0){return t[t.length-(1+e)]}function i(t){if(0===t.length)throw new Error("Invalid tail call");return[t.slice(0,t.length-1),t[t.length-1]]}function r(t,e,n=((t,e)=>t===e)){if(t===e)return!0;if(!t||!e)return!1;if(t.length!==e.length)return!1;for(let s=0,i=t.length;s<i;s++)if(!n(t[s],e[s]))return!1;return!0}function o(t,e){const n=t.length-1;e<n&&(t[e]=t[n]),t.pop()}function h(t,e,n){return function(t,e){let n=0,s=t-1;for(;n<=s;){const t=(n+s)/2|0,i=e(t);if(i<0)n=t+1;else{if(!(i>0))return t;s=t-1}}return-(n+1)}(t.length,(s=>n(t[s],e)))}function l(t,e,n){if((t|=0)>=e.length)throw new TypeError("invalid index");const s=e[Math.floor(e.length*Math.random())],i=[],r=[],o=[];for(const t of e){const e=n(t,s);e<0?i.push(t):e>0?r.push(t):o.push(t)}return t<i.length?l(t,i,n):t<i.length+o.length?o[0]:l(t-(i.length+o.length),r,n)}function a(t,e){const n=[];let s;for(const i of t.slice(0).sort(e))s&&0===e(s[0],i)?s.push(i):(s=[i],n.push(s));return n}function*c(t,e){let n,s;for(const i of t)void 0!==s&&e(s,i)?n.push(i):(n&&(yield n),n=[i]),s=i;n&&(yield n)}function u(t,e){for(let n=0;n<=t.length;n++)e(0===n?void 0:t[n-1],n===t.length?void 0:t[n])}function f(t,e){for(let n=0;n<t.length;n++)e(0===n?void 0:t[n-1],t[n],n+1===t.length?void 0:t[n+1])}function d(t){return t.filter((t=>!!t))}function _(t){let e=0;for(let n=0;n<t.length;n++)t[n]&&(t[e]=t[n],e+=1);t.length=e}function g(t){return!Array.isArray(t)||0===t.length}function p(t){return Array.isArray(t)&&t.length>0}function b(t,e=(t=>t)){const n=new Set;return t.filter((t=>{const s=e(t);return!n.has(s)&&(n.add(s),!0)}))}function v(t,e){return t.length>0?t[0]:e}function x(t,e){let n="number"==typeof e?t:0;"number"==typeof e?n=t:(n=0,e=t);const s=[];if(n<=e)for(let t=n;t<e;t++)s.push(t);else for(let t=n;t>e;t--)s.push(t);return s}function w(t,e,n){const s=t.slice(0,e),i=t.slice(e);return s.concat(n,i)}function y(t,e){const n=t.indexOf(e);n>-1&&(t.splice(n,1),t.unshift(e))}function I(t,e){const n=t.indexOf(e);n>-1&&(t.splice(n,1),t.push(e))}function m(t,e){for(const n of e)t.push(n)}function k(t){return Array.isArray(t)?t:[t]}function C(t,e,n,s){const i=L(t,e);let r=t.splice(i,n);return void 0===r&&(r=[]),function(t,e,n){const s=L(t,e),i=t.length,r=n.length;t.length=i+r;for(let e=i-1;e>=s;e--)t[e+r]=t[e];for(let e=0;e<r;e++)t[e+s]=n[e]}(t,i,s),r}function L(t,e){return e<0?Math.max(e+t.length,0):Math.min(e,t.length)}var M;function D(t,e){return(n,s)=>e(t(n),t(s))}function A(...t){return(e,n)=>{for(const s of t){const t=s(e,n);if(!M.isNeitherLessOrGreaterThan(t))return t}return M.neitherLessOrGreaterThan}}n.d(e,{BV:()=>W,EB:()=>b,Gb:()=>s,H9:()=>F,HW:()=>l,JH:()=>i,KO:()=>f,LS:()=>o,Of:()=>p,Rs:()=>_,W$:()=>O,XY:()=>g,Xh:()=>v,Zv:()=>w,_2:()=>k,_i:()=>R,al:()=>I,db:()=>C,fS:()=>r,f_:()=>A,fv:()=>T,kX:()=>d,mw:()=>c,nW:()=>E,ry:()=>h,tT:()=>D,vA:()=>m,vM:()=>a,w6:()=>x,zI:()=>y,zy:()=>u}),function(t){t.isLessThan=function(t){return t<0},t.isLessThanOrEqual=function(t){return t<=0},t.isGreaterThan=function(t){return t>0},t.isNeitherLessOrGreaterThan=function(t){return 0===t},t.greaterThan=1,t.lessThan=-1,t.neitherLessOrGreaterThan=0}(M||(M={}));const T=(t,e)=>t-e,E=(t,e)=>T(t?1:0,e?1:0);function W(t){return(e,n)=>-t(e,n)}class F{constructor(t){this.items=t,this.firstIdx=0,this.lastIdx=this.items.length-1}get length(){return this.lastIdx-this.firstIdx+1}takeWhile(t){let e=this.firstIdx;for(;e<this.items.length&&t(this.items[e]);)e++;const n=e===this.firstIdx?null:this.items.slice(this.firstIdx,e);return this.firstIdx=e,n}takeFromEndWhile(t){let e=this.lastIdx;for(;e>=0&&t(this.items[e]);)e--;const n=e===this.lastIdx?null:this.items.slice(e+1,this.lastIdx+1);return this.lastIdx=e,n}peek(){if(0!==this.length)return this.items[this.firstIdx]}dequeue(){const t=this.items[this.firstIdx];return this.firstIdx++,t}takeCount(t){const e=this.items.slice(this.firstIdx,this.firstIdx+t);return this.firstIdx+=t,e}}class O{constructor(t){this.iterate=t}toArray(){const t=[];return this.iterate((e=>(t.push(e),!0))),t}filter(t){return new O((e=>this.iterate((n=>!t(n)||e(n)))))}map(t){return new O((e=>this.iterate((n=>e(t(n))))))}findLast(t){let e;return this.iterate((n=>(t(n)&&(e=n),!0))),e}findLastMaxBy(t){let e,n=!0;return this.iterate((s=>((n||M.isGreaterThan(t(s,e)))&&(n=!1,e=s),!0))),e}}O.empty=new O((t=>{}));class R{constructor(t){this._indexMap=t}static createSortPermutation(t,e){const n=Array.from(t.keys()).sort(((n,s)=>e(t[n],t[s])));return new R(n)}apply(t){return t.map(((e,n)=>t[this._indexMap[n]]))}inverse(){const t=this._indexMap.slice();for(let e=0;e<this._indexMap.length;e++)t[this._indexMap[e]]=e;return new R(t)}}},35534:(t,e,n)=>{function s(t,e){const n=function(t,e,n=t.length-1){for(let s=n;s>=0;s--){if(e(t[s]))return s}return-1}(t,e);if(-1!==n)return t[n]}function i(t,e){const n=r(t,e);return-1===n?void 0:t[n]}function r(t,e,n=0,s=t.length){let i=n,r=s;for(;i<r;){const n=Math.floor((i+r)/2);e(t[n])?i=n+1:r=n}return i-1}function o(t,e){const n=h(t,e);return n===t.length?void 0:t[n]}function h(t,e,n=0,s=t.length){let i=n,r=s;for(;i<r;){const n=Math.floor((i+r)/2);e(t[n])?r=n:i=n+1}return i}n.d(e,{BS:()=>c,Fr:()=>d,J_:()=>h,Jw:()=>r,Y0:()=>f,b1:()=>l,cn:()=>o,dF:()=>s,hV:()=>a,ti:()=>i,vm:()=>u});class l{constructor(t){this._array=t,this._findLastMonotonousLastIdx=0}findLastMonotonous(t){if(l.assertInvariants){if(this._prevFindLastPredicate)for(const e of this._array)if(this._prevFindLastPredicate(e)&&!t(e))throw new Error("MonotonousArray: current predicate must be weaker than (or equal to) the previous predicate.");this._prevFindLastPredicate=t}const e=r(this._array,t,this._findLastMonotonousLastIdx);return this._findLastMonotonousLastIdx=e+1,-1===e?void 0:this._array[e]}}function a(t,e){if(0===t.length)return;let n=t[0];for(let s=1;s<t.length;s++){const i=t[s];e(i,n)>0&&(n=i)}return n}function c(t,e){if(0===t.length)return;let n=t[0];for(let s=1;s<t.length;s++){const i=t[s];e(i,n)>=0&&(n=i)}return n}function u(t,e){return a(t,((t,n)=>-e(t,n)))}function f(t,e){if(0===t.length)return-1;let n=0;for(let s=1;s<t.length;s++){e(t[s],t[n])>0&&(n=s)}return n}function d(t,e){for(const n of t){const t=e(n);if(void 0!==t)return t}}l.assertInvariants=!1},35146:(t,e,n)=>{n.d(e,{DM:()=>l,eZ:()=>h,ok:()=>i,vE:()=>r,wN:()=>o});var s=n(17301);function i(t,e){if(!t)throw new Error(e?`Assertion failed (${e})`:"Assertion Failed")}function r(t,e="Unreachable"){throw new Error(e)}function o(t){t||(0,s.dL)(new s.he("Soft Assertion Failed"))}function h(t){t()||(t(),(0,s.dL)(new s.he("Assertion Failed")))}function l(t,e){let n=0;for(;n<t.length-1;){if(!e(t[n],t[n+1]))return!1;n++}return!0}}}]);