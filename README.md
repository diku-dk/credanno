# MinAnno 🤏: Minimising annotation need in self-explanatory models

arXiv: 

Dataset: 

![intro](./imgs/intro.svg)

### Results table & anno_reduce figure

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;margin:0px auto;}
.tg td{background-color:rgba(255, 255, 255, 0.8);border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;
  word-break:normal;}
.tg th{background-color:rgba(240, 240, 240, 0.8);border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-yeaa{background-color:rgba(249, 249, 249, 0.8);border-color:inherit;font-weight:bold;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-jj9b{border-color:inherit;position:-webkit-sticky;position:sticky;text-align:left;top:-1px;vertical-align:middle;
  will-change:transform}
.tg .tg-ixdq{border-color:inherit;font-weight:bold;position:-webkit-sticky;position:sticky;text-align:center;top:-1px;
  vertical-align:middle;will-change:transform}
.tg .tg-pn43{background-color:rgba(249, 249, 249, 0.8);border-color:inherit;font-weight:bold;position:-webkit-sticky;position:sticky;
  text-align:center;top:-1px;vertical-align:middle;will-change:transform}
.tg .tg-kyy7{background-color:rgba(249, 249, 249, 0.8);border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-g7sd{border-color:inherit;font-weight:bold;text-align:left;vertical-align:middle}
.tg-sort-header::-moz-selection{background:0 0}
.tg-sort-header::selection{background:0 0}.tg-sort-header{cursor:pointer}
.tg-sort-header:after{content:'';float:right;margin-top:7px;border-width:0 5px 5px;border-style:solid;
  border-color:#404040 transparent;visibility:hidden}
.tg-sort-header:hover:after{visibility:visible}
.tg-sort-asc:after,.tg-sort-asc:hover:after,.tg-sort-desc:after{visibility:visible;opacity:.4}
.tg-sort-desc:after{border-bottom:none;border-width:5px 5px 0}</style>
<table id="tg-GbuUp" class="tg">
<thead>
  <tr>
    <th class="tg-jj9b" rowspan="2"></th>
    <th class="tg-ixdq" colspan="7">Nodule attributes</th>
    <th class="tg-ixdq" rowspan="2">Malignancy</th>
  </tr>
  <tr>
    <th class="tg-pn43">Sub</th>
    <th class="tg-pn43">Cal</th>
    <th class="tg-pn43">Sph</th>
    <th class="tg-pn43">Mar</th>
    <th class="tg-pn43">Lob</th>
    <th class="tg-pn43">Spi</th>
    <th class="tg-pn43">Tex</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-lboi" colspan="9">Full annotation</td>
  </tr>
  <tr>
    <td class="tg-yeaa">MinAnno (50-NN)</td>
    <td class="tg-kyy7">94.93</td>
    <td class="tg-kyy7">92.72</td>
    <td class="tg-kyy7">95.58</td>
    <td class="tg-kyy7">93.76</td>
    <td class="tg-kyy7">91.29</td>
    <td class="tg-kyy7">92.72</td>
    <td class="tg-kyy7">94.67</td>
    <td class="tg-kyy7">87.52</td>
  </tr>
  <tr>
    <td class="tg-g7sd">MinAnno (250-NN)</td>
    <td class="tg-9wq8">96.36</td>
    <td class="tg-9wq8">92.59</td>
    <td class="tg-9wq8">96.23</td>
    <td class="tg-9wq8">94.15</td>
    <td class="tg-9wq8">90.90</td>
    <td class="tg-9wq8">92.33</td>
    <td class="tg-9wq8">92.72</td>
    <td class="tg-9wq8">88.95</td>
  </tr>
  <tr>
    <td class="tg-yeaa">MinAnno (trained)</td>
    <td class="tg-kyy7">95.84</td>
    <td class="tg-kyy7">95.97</td>
    <td class="tg-kyy7">97.40</td>
    <td class="tg-kyy7">96.49</td>
    <td class="tg-kyy7">94.15</td>
    <td class="tg-kyy7">94.41</td>
    <td class="tg-kyy7">97.01</td>
    <td class="tg-kyy7">88.30</td>
  </tr>
  <tr>
    <td class="tg-lboi" colspan="9">Partial annotation</td>
  </tr>
  <tr>
    <td class="tg-yeaa">MinAnno (10%, 50-NN)</td>
    <td class="tg-kyy7">94.93</td>
    <td class="tg-kyy7">92.07</td>
    <td class="tg-kyy7">96.75</td>
    <td class="tg-kyy7">94.28</td>
    <td class="tg-kyy7">92.59</td>
    <td class="tg-kyy7">91.16</td>
    <td class="tg-kyy7">94.15</td>
    <td class="tg-kyy7">87.13</td>
  </tr>
  <tr>
    <td class="tg-g7sd">MinAnno (10%, 150-NN)</td>
    <td class="tg-9wq8">95.32</td>
    <td class="tg-9wq8">89.47</td>
    <td class="tg-9wq8">97.01</td>
    <td class="tg-9wq8">93.89</td>
    <td class="tg-9wq8">91.81</td>
    <td class="tg-9wq8">90.51</td>
    <td class="tg-9wq8">92.85</td>
    <td class="tg-9wq8">88.17</td>
  </tr>
  <tr>
    <td class="tg-yeaa">MinAnno (1%, trained) 🤏</td>
    <td class="tg-kyy7">91.81</td>
    <td class="tg-kyy7">93.37</td>
    <td class="tg-kyy7">96.49</td>
    <td class="tg-kyy7">90.77</td>
    <td class="tg-kyy7">89.73</td>
    <td class="tg-kyy7">92.33</td>
    <td class="tg-kyy7">93.76</td>
    <td class="tg-kyy7">86.09</td>
  </tr>
</tbody>
</table>
<script charset="utf-8">var TGSort=window.TGSort||function(n){"use strict";function r(n){return n?n.length:0}function t(n,t,e,o=0){for(e=r(n);o<e;++o)t(n[o],o)}function e(n){return n.split("").reverse().join("")}function o(n){var e=n[0];return t(n,function(n){for(;!n.startsWith(e);)e=e.substring(0,r(e)-1)}),r(e)}function u(n,r,e=[]){return t(n,function(n){r(n)&&e.push(n)}),e}var a=parseFloat;function i(n,r){return function(t){var e="";return t.replace(n,function(n,t,o){return e=t.replace(r,"")+"."+(o||"").substring(1)}),a(e)}}var s=i(/^(?:\s*)([+-]?(?:\d+)(?:,\d{3})*)(\.\d*)?$/g,/,/g),c=i(/^(?:\s*)([+-]?(?:\d+)(?:\.\d{3})*)(,\d*)?$/g,/\./g);function f(n){var t=a(n);return!isNaN(t)&&r(""+t)+1>=r(n)?t:NaN}function d(n){var e=[],o=n;return t([f,s,c],function(u){var a=[],i=[];t(n,function(n,r){r=u(n),a.push(r),r||i.push(n)}),r(i)<r(o)&&(o=i,e=a)}),r(u(o,function(n){return n==o[0]}))==r(o)?e:[]}function v(n){if("TABLE"==n.nodeName){for(var a=function(r){var e,o,u=[],a=[];return function n(r,e){e(r),t(r.childNodes,function(r){n(r,e)})}(n,function(n){"TR"==(o=n.nodeName)?(e=[],u.push(e),a.push(n)):"TD"!=o&&"TH"!=o||e.push(n)}),[u,a]}(),i=a[0],s=a[1],c=r(i),f=c>1&&r(i[0])<r(i[1])?1:0,v=f+1,p=i[f],h=r(p),l=[],g=[],N=[],m=v;m<c;++m){for(var T=0;T<h;++T){r(g)<h&&g.push([]);var C=i[m][T],L=C.textContent||C.innerText||"";g[T].push(L.trim())}N.push(m-v)}t(p,function(n,t){l[t]=0;var a=n.classList;a.add("tg-sort-header"),n.addEventListener("click",function(){var n=l[t];!function(){for(var n=0;n<h;++n){var r=p[n].classList;r.remove("tg-sort-asc"),r.remove("tg-sort-desc"),l[n]=0}}(),(n=1==n?-1:+!n)&&a.add(n>0?"tg-sort-asc":"tg-sort-desc"),l[t]=n;var i,f=g[t],m=function(r,t){return n*f[r].localeCompare(f[t])||n*(r-t)},T=function(n){var t=d(n);if(!r(t)){var u=o(n),a=o(n.map(e));t=d(n.map(function(n){return n.substring(u,r(n)-a)}))}return t}(f);(r(T)||r(T=r(u(i=f.map(Date.parse),isNaN))?[]:i))&&(m=function(r,t){var e=T[r],o=T[t],u=isNaN(e),a=isNaN(o);return u&&a?0:u?-n:a?n:e>o?n:e<o?-n:n*(r-t)});var C,L=N.slice();L.sort(m);for(var E=v;E<c;++E)(C=s[E].parentNode).removeChild(s[E]);for(E=v;E<c;++E)C.appendChild(s[v+L[E-v]])})})}}n.addEventListener("DOMContentLoaded",function(){for(var t=n.getElementsByClassName("tg"),e=0;e<r(t);++e)try{v(t[e])}catch(n){}})}(document)</script>

## Dependencies


## Usage instruction

### Data pre-processing

### Training

### Evaluation

### Pretrained model



