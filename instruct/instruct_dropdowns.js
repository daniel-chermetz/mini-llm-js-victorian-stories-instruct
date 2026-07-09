// instruct_dropdowns.js
// -----------------------------------------------------------------------------
// Builds the "Step 1" prompt-builder UI: one dropdown per story-element
// category. Categories and their elements come from
//   ./instruct/shortened_to_flattened_lower.json
// where each key is a category name and each value is an array of elements
// shaped like [ "Display Name", [ ...flattened tokens... ] ]. The zero-index
// string of every element is used as the human-readable option label.
//
// The user may make up to MAX_SELECTIONS picks (at most one per dropdown),
// with OPTIMAL_SELECTIONS advertised as the sweet spot. Selections are shown
// as removable chips (each with an "X" to cancel it).
//
// No build step required: this file registers itself on globalThis and is
// loaded via a plain <script> tag. Call InstructBuilder.init(container) once
// the DOM is ready.
// -----------------------------------------------------------------------------
(function () {
    'use strict';

    const DATA_URL = './instruct/shortened_to_flattened_lower.json';
    // Per-category metadata: value[0] = priority (sort order), value[2] = the
    // tokenized category name used before the ">" separator in the prompt.
    const KEYS_URL = './instruct/element_keys_tokenized_dual.json';
    const MAX_SELECTIONS = 6;
    const OPTIMAL_SELECTIONS = 4;

    // Structural tokens that wrap the instruct prompt. Each is an exact entry in
    // the model vocabulary, mirroring the format produced on the training side.
    const TOK_OPEN = '[';
    const TOK_CLOSE = ']';
    const TOK_PLUS = '+';
    const TOK_ARROW = '>';
    const TOK_NEWLINE = '\n';

    const state = {
        data: null,            // { category: [ [displayName, tokens], ... ], ... }
        keys: null,            // { category: [ priority, [nameTokensTitle], [nameTokens] ] }
        container: null,       // root element the UI is rendered into
        selections: new Map(), // category -> { name, tokens }
        selectEls: new Map(),  // category -> <select> element
        chipsEl: null,
        counterEl: null,
        gridEl: null,
        continueBtn: null,
        onContinue: null,      // callback(tokens, selections) invoked on "Continue"
    };

    // ------------------------------------------------------------------ styles
    function injectStyles() {
        if (document.getElementById('instruct-dropdowns-styles')) return;
        const style = document.createElement('style');
        style.id = 'instruct-dropdowns-styles';
        style.textContent = `
            .instruct-builder h2 {
                text-align: center;
                font-weight: 600;
                color: var(--text-color, #1e293b);
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 1.5em;
                letter-spacing: -0.01em;
            }
            .instruct-advisory {
                text-align: center;
                color: var(--text-muted, #64748b);
                font-size: 0.9em;
                margin: 0 0 18px;
            }
            .instruct-counter {
                text-align: center;
                font-weight: 600;
                font-size: 0.9em;
                margin-bottom: 16px;
                color: var(--text-muted, #64748b);
            }
            .instruct-counter.is-optimal { color: var(--success-dark, #059669); }
            .instruct-counter.is-full { color: var(--primary-dark, #4f46e5); }

            .instruct-chips {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                min-height: 8px;
                margin-bottom: 22px;
            }
            .instruct-chips:empty { margin-bottom: 0; }
            .instruct-chip {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: linear-gradient(135deg, var(--primary-color, #6366f1) 0%, var(--primary-dark, #4f46e5) 100%);
                color: #fff;
                border-radius: 999px;
                padding: 7px 8px 7px 14px;
                font-size: 0.88em;
                font-weight: 500;
                box-shadow: var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.1));
                animation: instructChipIn 0.15s ease-out;
            }
            @keyframes instructChipIn {
                from { opacity: 0; transform: scale(0.9); }
                to   { opacity: 1; transform: scale(1); }
            }
            .instruct-chip .instruct-chip-cat {
                opacity: 0.85;
                font-weight: 400;
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            .instruct-chip-remove {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 20px;
                height: 20px;
                border: none;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.22);
                color: #fff;
                font-size: 14px;
                line-height: 1;
                cursor: pointer;
                transition: background 0.15s ease;
            }
            .instruct-chip-remove:hover { background: rgba(255, 255, 255, 0.45); }

            .instruct-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
                gap: 16px 20px;
            }
            .instruct-field { display: flex; flex-direction: column; gap: 6px; }
            .instruct-field-label {
                font-size: 0.8em;
                font-weight: 600;
                color: var(--text-muted, #64748b);
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }
            .instruct-select {
                width: 100%;
                padding: 10px 12px;
                font-family: var(--font-family, inherit);
                font-size: 0.95em;
                color: var(--text-color, #1e293b);
                background-color: #fff;
                border: 2px solid var(--border-color, #e2e8f0);
                border-radius: 10px;
                cursor: pointer;
                transition: border-color 0.15s ease, box-shadow 0.15s ease;
                box-sizing: border-box;
            }
            .instruct-select:hover:not(:disabled) { border-color: var(--primary-light, #818cf8); }
            .instruct-select:focus {
                outline: none;
                border-color: var(--primary-color, #6366f1);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
            }
            .instruct-select.has-value {
                border-color: var(--primary-color, #6366f1);
                font-weight: 600;
            }
            .instruct-select:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                background-color: #f8fafc;
            }

            .instruct-actions {
                display: flex;
                justify-content: center;
                margin: 4px 0 24px;
            }
            .instruct-continue-btn {
                padding: 14px 40px;
                font-family: var(--font-family, inherit);
                font-size: 1.05em;
                font-weight: 600;
                color: #fff;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                background: linear-gradient(135deg, var(--success-color, #10b981) 0%, var(--success-dark, #059669) 100%);
                box-shadow: var(--shadow-md, 0 4px 6px rgba(0,0,0,0.1));
                transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
            }
            .instruct-continue-btn:hover:not(:disabled) {
                transform: translateY(-1px);
                box-shadow: var(--shadow-lg, 0 10px 15px rgba(0,0,0,0.1));
            }
            .instruct-continue-btn:disabled {
                opacity: 0.45;
                cursor: not-allowed;
                background: var(--secondary-color, #64748b);
            }
        `;
        document.head.appendChild(style);
    }

    // -------------------------------------------------------------- data load
    async function loadData() {
        if (state.data && state.keys) return state.data;
        const [dataRes, keysRes] = await Promise.all([fetch(DATA_URL), fetch(KEYS_URL)]);
        if (!dataRes.ok) {
            throw new Error(`Failed to load ${DATA_URL} (HTTP ${dataRes.status})`);
        }
        if (!keysRes.ok) {
            throw new Error(`Failed to load ${KEYS_URL} (HTTP ${keysRes.status})`);
        }
        state.data = await dataRes.json();
        state.keys = await keysRes.json();
        return state.data;
    }

    // ------------------------------------------------------------------- build
    function buildUI() {
        const root = state.container;
        root.innerHTML = '';
        root.classList.add('instruct-builder');

        const heading = document.createElement('h2');
        heading.textContent = 'Step 1: Build Your Story Prompt';
        root.appendChild(heading);

        const advisory = document.createElement('p');
        advisory.className = 'instruct-advisory';
        advisory.textContent =
            `Pick from the categories below — up to ${MAX_SELECTIONS} in total ` +
            `(one per category). Around ${OPTIMAL_SELECTIONS} selections tends to work best.`;
        root.appendChild(advisory);

        const counter = document.createElement('div');
        counter.className = 'instruct-counter';
        state.counterEl = counter;
        root.appendChild(counter);

        const chips = document.createElement('div');
        chips.className = 'instruct-chips';
        state.chipsEl = chips;
        root.appendChild(chips);

        // Continue button sits above the category grid so it's visible without
        // scrolling. It stays disabled until at least one selection is made.
        const actions = document.createElement('div');
        actions.className = 'instruct-actions';
        const continueBtn = document.createElement('button');
        continueBtn.type = 'button';
        continueBtn.className = 'instruct-continue-btn';
        continueBtn.textContent = 'Continue to Generate';
        continueBtn.addEventListener('click', onContinueClick);
        state.continueBtn = continueBtn;
        actions.appendChild(continueBtn);
        root.appendChild(actions);

        const grid = document.createElement('div');
        grid.className = 'instruct-grid';
        state.gridEl = grid;
        root.appendChild(grid);

        for (const category of Object.keys(state.data)) {
            grid.appendChild(buildField(category, state.data[category]));
        }

        renderChips();
        updateCounter();
        updateDisabledState();
    }

    function buildField(category, elements) {
        const field = document.createElement('div');
        field.className = 'instruct-field';

        const label = document.createElement('label');
        label.className = 'instruct-field-label';
        label.textContent = category;

        const select = document.createElement('select');
        select.className = 'instruct-select';
        select.dataset.category = category;

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = '— Choose —';
        select.appendChild(placeholder);

        elements.forEach((element, index) => {
            const option = document.createElement('option');
            option.value = String(index);
            option.textContent = Array.isArray(element) ? element[0] : String(element);
            select.appendChild(option);
        });

        const selectId = 'instruct-select-' + slug(category);
        select.id = selectId;
        label.setAttribute('for', selectId);

        select.addEventListener('change', () => onSelectChange(category, select));

        state.selectEls.set(category, select);

        field.appendChild(label);
        field.appendChild(select);
        return field;
    }

    // ------------------------------------------------------------- selection
    function onSelectChange(category, select) {
        const value = select.value;
        if (value === '') {
            state.selections.delete(category);
        } else {
            const element = state.data[category][Number(value)];
            state.selections.set(category, {
                name: Array.isArray(element) ? element[0] : String(element),
                tokens: Array.isArray(element) ? element[1] : null,
            });
        }
        select.classList.toggle('has-value', value !== '');
        renderChips();
        updateCounter();
        updateDisabledState();
    }

    function removeSelection(category) {
        state.selections.delete(category);
        const select = state.selectEls.get(category);
        if (select) {
            select.value = '';
            select.classList.remove('has-value');
        }
        renderChips();
        updateCounter();
        updateDisabledState();
    }

    // --------------------------------------------------------------- render
    function renderChips() {
        const chips = state.chipsEl;
        chips.innerHTML = '';
        for (const [category, element] of state.selections) {
            const chip = document.createElement('span');
            chip.className = 'instruct-chip';

            const cat = document.createElement('span');
            cat.className = 'instruct-chip-cat';
            cat.textContent = category;

            const name = document.createElement('span');
            name.textContent = element.name;

            const remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'instruct-chip-remove';
            remove.setAttribute('aria-label', `Remove ${element.name}`);
            remove.textContent = '\u00d7'; // ×
            remove.addEventListener('click', () => removeSelection(category));

            chip.appendChild(cat);
            chip.appendChild(name);
            chip.appendChild(remove);
            chips.appendChild(chip);
        }
    }

    function updateCounter() {
        const count = state.selections.size;
        const counter = state.counterEl;
        counter.textContent = `${count} of ${MAX_SELECTIONS} selected`;
        counter.classList.toggle('is-full', count >= MAX_SELECTIONS);
        counter.classList.toggle(
            'is-optimal',
            count >= OPTIMAL_SELECTIONS && count < MAX_SELECTIONS
        );
        if (state.continueBtn) {
            state.continueBtn.disabled = count === 0;
        }
    }

    // When the cap is reached, lock the dropdowns that have no pick yet so the
    // user can't exceed MAX_SELECTIONS. They unlock again after a removal.
    function updateDisabledState() {
        const full = state.selections.size >= MAX_SELECTIONS;
        for (const [category, select] of state.selectEls) {
            const isSelected = state.selections.has(category);
            select.disabled = full && !isSelected;
        }
    }

    // --------------------------------------------------------------- helpers
    function slug(text) {
        return String(text).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    }

    function getPriority(category) {
        const meta = state.keys && state.keys[category];
        // Missing metadata sorts last but never breaks the build.
        return meta && Number.isFinite(meta[0]) ? meta[0] : Number.MAX_SAFE_INTEGER;
    }

    function getCategoryNameTokens(category) {
        const meta = state.keys && state.keys[category];
        // Index 2 holds the tokenized (lower-case) category name.
        if (meta && Array.isArray(meta[2])) return meta[2];
        // Fallback: emit the raw category string as a single token.
        return [category];
    }

    // Assemble the flat instruct-prompt token array from the current selections,
    // ordered by category priority. Mirrors the training-side format:
    //   [ [ <catTokens> > <valueTokens> ] + [ ... ] ] \n \n
    function buildInstructTokens() {
        const picks = Array.from(state.selections.entries())
            .map(([category, element]) => ({
                category,
                priority: getPriority(category),
                nameTokens: getCategoryNameTokens(category),
                valueTokens: Array.isArray(element.tokens) ? element.tokens : [],
            }))
            .sort((a, b) => a.priority - b.priority);

        const tokens = [TOK_OPEN];
        picks.forEach((pick, index) => {
            if (index > 0) tokens.push(TOK_PLUS);
            tokens.push(TOK_OPEN);
            for (const t of pick.nameTokens) tokens.push(t);
            tokens.push(TOK_ARROW);
            for (const t of pick.valueTokens) tokens.push(t);
            tokens.push(TOK_CLOSE);
        });
        tokens.push(TOK_CLOSE);
        tokens.push(TOK_NEWLINE, TOK_NEWLINE);
        return tokens;
    }

    function onContinueClick() {
        if (state.selections.size === 0) return;
        const tokens = buildInstructTokens();
        if (typeof state.onContinue === 'function') {
            state.onContinue(tokens, api.getSelections());
        }
    }

    // ------------------------------------------------------------------- api
    async function init(container, options) {
        state.container =
            typeof container === 'string' ? document.querySelector(container) : container;
        if (!state.container) {
            throw new Error('InstructBuilder.init: container not found');
        }
        if (options && typeof options.onContinue === 'function') {
            state.onContinue = options.onContinue;
        }
        await loadData();
        injectStyles();
        buildUI();
        return api;
    }

    const api = {
        init,
        // Returns the current picks as [{ category, name, tokens }, ...].
        getSelections() {
            return Array.from(state.selections.entries()).map(([category, element]) => ({
                category,
                name: element.name,
                tokens: element.tokens,
            }));
        },
        // Returns the flat instruct-prompt token array for the current picks.
        buildInstructTokens,
        clear() {
            for (const category of Array.from(state.selections.keys())) {
                removeSelection(category);
            }
        },
        MAX_SELECTIONS,
        OPTIMAL_SELECTIONS,
    };

    globalThis.InstructBuilder = api;
})();
