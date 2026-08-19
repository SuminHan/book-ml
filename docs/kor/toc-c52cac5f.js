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
        this.innerHTML = '<ol class="chapter"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="index.html">Introduction</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 1</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter01.html"><strong aria-hidden="true">1.</strong> Orientation</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">1.1.</strong> 1.1 머신러닝 문제 정식화와 세 갈래 분류</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">1.2.</strong> 1.2 데이터사이언스 파이프라인과 사전지식 리뷰</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">1.3.</strong> 1.3 아주 짧은 역사와 이번 학기 로드맵</span></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter02.html"><strong aria-hidden="true">2.</strong> 회귀 모델: 선형회귀와 로지스틱회귀 (Regression Models: Linear &amp; Logistic)</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">2.1.</strong> 2.1 선형회귀: 모델, 비용함수, 경사하강법</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">2.2.</strong> 2.2 정규방정식과 &quot;같은 모델, 다른 출력&quot;이라는 다리</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">2.3.</strong> 2.3 로지스틱회귀: 시그모이드에서 PR-AUC까지</span></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter03.html"><strong aria-hidden="true">3.</strong> 생성 모델 관점의 분류: 나이브베이즈와 GDA (Generative Classifiers: Naive Bayes &amp; GDA)</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">3.1.</strong> 3.1 베이즈 정리, 그리고 생성 vs 판별</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">3.2.</strong> 3.2 가우시안 판별분석 (GDA)</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">3.3.</strong> 3.3 나이브베이즈: 독립을 가정하고 차원의 저주를 피한다</span></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter04.html"><strong aria-hidden="true">4.</strong> 거리 기반 모델과 클러스터링: kNN과 k-means (Distance-Based Models &amp; Clustering)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter05.html"><strong aria-hidden="true">5.</strong> SVM과 커널 (Support Vector Machines &amp; Kernels)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter06.html"><strong aria-hidden="true">6.</strong> 정규화와 모델 선택 (Regularization &amp; Model Selection)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter07.html"><strong aria-hidden="true">7.</strong> 트리 기반 모델: 결정트리에서 GBDT까지 (Tree-Based Models: Decision Trees to GBDT)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter08.html"><strong aria-hidden="true">8.</strong> Block A 캡스톤: 팀 프로젝트와 총정리 (Block A Capstone)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter09.html"><strong aria-hidden="true">9.</strong> 신경망 기초, 역전파, 학습 기법 (Neural Networks, Backprop &amp; Training)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter10.html"><strong aria-hidden="true">10.</strong> CNN 기초와 응용 (CNN Basics &amp; Applications)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter11.html"><strong aria-hidden="true">11.</strong> 시퀀스 모델 (Sequence Models)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter12.html"><strong aria-hidden="true">12.</strong> 어텐션과 트랜스포머 (Attention &amp; Transformer)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter13.html"><strong aria-hidden="true">13.</strong> LLM: 사전학습, 프롬프팅, 정렬 (LLM: Pretraining, Prompting &amp; Alignment)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter14.html"><strong aria-hidden="true">14.</strong> 표현학습: PCA, word2vec, Node2Vec, PageRank (Representation Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter15.html"><strong aria-hidden="true">15.</strong> 잠재변수 생성모델: EM/GMM에서 VAE, GAN, Diffusion까지 (Latent-Variable Generative Models)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter16.html"><strong aria-hidden="true">16.</strong> Block B 캡스톤: 팀 프로젝트와 ML1 총정리 (Block B Capstone &amp; ML1 Review)</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 2</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter01.html"><strong aria-hidden="true">17.</strong> 코스 소개와 신경망 미니 리뷰 (Course Introduction &amp; Neural Network Mini-Review)</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">17.1.</strong> 1.1 강화학습이란 무엇인가, 그리고 이번 학기 로드맵</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">17.2.</strong> 1.2 신경망 미니 리뷰</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">17.3.</strong> 1.3 실습 환경 세팅: Gymnasium</span></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter02.html"><strong aria-hidden="true">18.</strong> 멀티암 밴딧 (Multi-Armed Bandits)</a><a class="chapter-fold-toggle"><div>❱</div></a></span><ol class="section"><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">18.1.</strong> 2.1 탐험과 활용: ε-greedy</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">18.2.</strong> 2.2 낙관적 초기화와 UCB: 불확실성을 이용한 탐험</span></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><span><strong aria-hidden="true">18.3.</strong> 2.3 밴딧 vs 완전한 MDP: 다음 장으로 가는 다리</span></span></li></ol><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter03.html"><strong aria-hidden="true">19.</strong> MDP 정식화 (Markov Decision Processes)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter04.html"><strong aria-hidden="true">20.</strong> 동적계획법 (Dynamic Programming)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter05.html"><strong aria-hidden="true">21.</strong> 몬테카를로 방법 (Monte Carlo Methods)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter06.html"><strong aria-hidden="true">22.</strong> 시간차 학습 (Temporal-Difference Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter07.html"><strong aria-hidden="true">23.</strong> n-step 부트스트래핑, 적격흔적, 그리고 계획 (n-Step Bootstrapping, Eligibility Traces &amp; Planning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter08.html"><strong aria-hidden="true">24.</strong> Block A 캡스톤: 팀 프로젝트와 총정리 (Block A Capstone)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter09.html"><strong aria-hidden="true">25.</strong> 함수근사와 DQN (Function Approximation &amp; Deep Q-Networks)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter10.html"><strong aria-hidden="true">26.</strong> 정책기반 강화학습 (Policy-Based Reinforcement Learning)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter11.html"><strong aria-hidden="true">27.</strong> 고급 정책 최적화: PPO (Proximal Policy Optimization)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter12.html"><strong aria-hidden="true">28.</strong> 모방학습과 인간 피드백 (Imitation Learning &amp; Learning from Human Feedback)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter13.html"><strong aria-hidden="true">29.</strong> 로봇 시뮬레이션과 제어 기초 (Robot Simulation &amp; Control Basics)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter14.html"><strong aria-hidden="true">30.</strong> 고급 시뮬레이션: MuJoCo와 Isaac Sim (Advanced Simulation: MuJoCo &amp; Isaac Sim)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter15.html"><strong aria-hidden="true">31.</strong> 모델기반 RL과 몬테카를로 트리 탐색 (Model-Based RL &amp; Monte Carlo Tree Search)</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter16.html"><strong aria-hidden="true">32.</strong> Block B 캡스톤: 팀 프로젝트와 학기 총정리 (Block B Capstone &amp; Semester Review)</a></span></li></ol>';
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

