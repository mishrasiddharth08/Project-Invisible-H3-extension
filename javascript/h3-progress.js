// Reuse Forge's polling and overall bar; add only the current H3 image bar.
onUiLoaded(function () {
    if (typeof requestProgress !== 'function' || requestProgress._h3Dual) return;
    const original = requestProgress;
    requestProgress = function (id, container, gallery, atEnd, onProgress, timeout) {
        let current = null, overall = null;
        const previewGallery = gallery && gallery.classList.contains('hidden')
            ? gallery.parentElement.querySelector('.gradio-video') : gallery;
        const clear = () => {
            if (current) current.remove();
            container.parentNode.classList.remove('pi-h3-dual-progress');
            current = null; overall = null;
        };
        return original(id, container, gallery, function () {
            if (previewGallery) previewGallery.classList.remove('pi-h3-developing');
            clear(); if (atEnd) atEnd();
        }, function (res) {
            const match = /^H3: image (\d+)\/(\d+) \|/.exec(res.textinfo || '');
            if (previewGallery) previewGallery.classList.toggle('pi-h3-developing', !!match && !!res.active);
            if (match && res.active && opts.show_progressbar) {
                if (!current) {
                    overall = container.previousElementSibling;
                    current = document.createElement('div');
                    current.className = 'progressDiv pi-h3-current';
                    current.appendChild(document.createElement('div')).className = 'progress';
                    container.parentNode.insertBefore(current, overall);
                    container.parentNode.classList.add('pi-h3-dual-progress');
                }
                const index = Number(match[1]), total = Number(match[2]);
                const fraction = Math.max(0, Math.min(1, (res.progress || 0) * total - (index - 1)));
                const inner = current.firstElementChild;
                inner.style.width = (fraction * 100) + '%';
                inner.textContent = `Current image ${index}/${total}: ${Math.round(fraction * 100)}%`;
                const native = overall && overall.querySelector('.progress');
                if (native) native.textContent = `Overall: ${Math.round((res.progress || 0) * 100)}% — ${res.textinfo.split(' | ')[1]}`;
            } else clear();
            if (onProgress) onProgress(res);
        }, timeout);
    };
    requestProgress._h3Dual = true;
});
