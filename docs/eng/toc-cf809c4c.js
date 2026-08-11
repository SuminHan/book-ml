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
        this.innerHTML = '<ol class="chapter"><li class="chapter-item "><span class="chapter-link-wrapper"><a href="index.html">Introduction</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 1</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter01.html"><strong aria-hidden="true">1.</strong> Orientation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter02.html"><strong aria-hidden="true">2.</strong> Linear Regression</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter03.html"><strong aria-hidden="true">3.</strong> Logistic Regression &amp; Classification Metrics</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter04.html"><strong aria-hidden="true">4.</strong> Distance-Based Models: kNN</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter05.html"><strong aria-hidden="true">5.</strong> Tree-Based Models</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter06.html"><strong aria-hidden="true">6.</strong> GBDT &amp; Explainability</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter07.html"><strong aria-hidden="true">7.</strong> Neural Network Basics &amp; Backpropagation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter08.html"><strong aria-hidden="true">8.</strong> Deep Learning Training Techniques</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter09.html"><strong aria-hidden="true">9.</strong> CNN Basics</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter10.html"><strong aria-hidden="true">10.</strong> Unsupervised Learning &amp; Representation Learning</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter11.html"><strong aria-hidden="true">11.</strong> Reinforcement Learning Preview</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml1/chapter12.html"><strong aria-hidden="true">12.</strong> Generative Models Preview &amp; Review</a></span></li><li class="chapter-item "><li class="part-title">Machine Learning 2</li></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter01.html"><strong aria-hidden="true">13.</strong> ML1 Review &amp; Roadmap</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter02.html"><strong aria-hidden="true">14.</strong> Sequence Models</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter03.html"><strong aria-hidden="true">15.</strong> Attention &amp; Transformer</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter04.html"><strong aria-hidden="true">16.</strong> LLM Preview</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter05.html"><strong aria-hidden="true">17.</strong> Reinforcement Learning Basics &amp; Policy Evaluation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter06.html"><strong aria-hidden="true">18.</strong> Reinforcement Learning Algorithms</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter07.html"><strong aria-hidden="true">19.</strong> Deep Reinforcement Learning</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter08.html"><strong aria-hidden="true">20.</strong> Policy-Based Reinforcement Learning</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter09.html"><strong aria-hidden="true">21.</strong> Generative Models I: Likelihood-Based</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter10.html"><strong aria-hidden="true">22.</strong> Generative Models II: Adversarial &amp; Score-Based</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter11.html"><strong aria-hidden="true">23.</strong> Team Project: Implementation</a></span></li><li class="chapter-item "><span class="chapter-link-wrapper"><a href="ml2/chapter12.html"><strong aria-hidden="true">24.</strong> Team Project: Presentation</a></span></li></ol>';
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

