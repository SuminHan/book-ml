// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="index.html">Introduction</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 1</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter01.html"><strong aria-hidden="true">1.</strong> Orientation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter02.html"><strong aria-hidden="true">2.</strong> 회귀 모델: 선형회귀와 로지스틱회귀 (Regression Models: Linear &amp; Logistic)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter03.html"><strong aria-hidden="true">3.</strong> 생성 모델 관점의 분류: 나이브베이즈와 GDA (Generative Classifiers: Naive Bayes &amp; GDA)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter04.html"><strong aria-hidden="true">4.</strong> 거리 기반 모델과 클러스터링: kNN과 k-means (Distance-Based Models &amp; Clustering)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter05.html"><strong aria-hidden="true">5.</strong> SVM과 커널 (Support Vector Machines &amp; Kernels)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter06.html"><strong aria-hidden="true">6.</strong> 정규화와 모델 선택 (Regularization &amp; Model Selection)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter07.html"><strong aria-hidden="true">7.</strong> 트리 기반 모델: 결정트리에서 GBDT까지 (Tree-Based Models: Decision Trees to GBDT)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter08.html"><strong aria-hidden="true">8.</strong> 신경망 기초와 역전파 (Neural Network Basics &amp; Backpropagation)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter09.html"><strong aria-hidden="true">9.</strong> 딥러닝 학습 기법 (Deep Learning Training Techniques)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter10.html"><strong aria-hidden="true">10.</strong> CNN 기초 (CNN Basics)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter11.html"><strong aria-hidden="true">11.</strong> CNN 응용과 최신 아키텍처 (CNN Applications &amp; Modern Architectures)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter12.html"><strong aria-hidden="true">12.</strong> 비지도학습과 표현학습 (Unsupervised Learning &amp; Representation Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter13.html"><strong aria-hidden="true">13.</strong> EM 알고리즘과 가우시안 혼합모델 (EM Algorithm &amp; Gaussian Mixture Models)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter14.html"><strong aria-hidden="true">14.</strong> 생성형 모델 맛보기와 총정리 (Generative Models Preview &amp; Review)</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 2</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter01.html"><strong aria-hidden="true">15.</strong> ML1 복습과 로드맵 (ML1 Review &amp; Roadmap)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter02.html"><strong aria-hidden="true">16.</strong> 시퀀스 모델 (Sequence Models)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter03.html"><strong aria-hidden="true">17.</strong> 어텐션과 트랜스포머 (Attention &amp; Transformer)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter04.html"><strong aria-hidden="true">18.</strong> LLM 맛보기 (LLM Preview)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter05.html"><strong aria-hidden="true">19.</strong> 강화학습 기초와 정책평가 (Reinforcement Learning Basics &amp; Policy Evaluation)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter06.html"><strong aria-hidden="true">20.</strong> 강화학습 알고리즘 (Reinforcement Learning Algorithms)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter07.html"><strong aria-hidden="true">21.</strong> 심층강화학습 (Deep Reinforcement Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter08.html"><strong aria-hidden="true">22.</strong> 정책기반 강화학습 (Policy-Based Reinforcement Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter09.html"><strong aria-hidden="true">23.</strong> 생성형 모델 I: 우도 기반 (Generative Models I: Likelihood-Based)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter10.html"><strong aria-hidden="true">24.</strong> 생성형 모델 II: 적대적/스코어 기반 (Generative Models II: Adversarial &amp; Score-Based)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter11.html"><strong aria-hidden="true">25.</strong> 팀 프로젝트: 구현 (Team Project: Implementation)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter12.html"><strong aria-hidden="true">26.</strong> 팀 프로젝트: 발표 (Team Project: Presentation)</a></span></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            // Check both with and without the '.html' suffix to be robust against pretty URLs
            if (link.href.replace(/\.html$/, '') === current_page.replace(/\.html$/, '')
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);

