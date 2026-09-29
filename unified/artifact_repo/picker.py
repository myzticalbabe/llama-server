"""tkinter GUI model picker + artifact repository browser."""
import os, sys, tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional

# ── Setup path to find unified modules ─────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from artifact_repo.repository import ArtifactRepository
from artifact_repo.database import DB_PATH


class ArtifactPicker:
    """Desktop model picker with artifact repository browser."""

    def __init__(self, repo: Optional[ArtifactRepository] = None):
        self._repo = repo or ArtifactRepository()
        self._current_model: Optional[dict] = None
        self._sort_by = tk.StringVar(value='updated')
        self._search_var = tk.StringVar()
        self._tag_var = tk.StringVar()
        self._build_ui()

    def _build_ui(self):
        self._win = tk.Tk()
        self._win.title("Artifact Repository — Model Picker")
        self._win.geometry("1100x700")
        self._win.minsize(800, 500)

        # Try to use modern theme
        style = ttk.Style()
        try:
            style.theme_use('vista')
        except tk.TclError:
            pass

        # ── Top toolbar ────────────────────────────────────────────────
        toolbar = ttk.Frame(self._win, padding=5)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        search_entry = ttk.Entry(toolbar, textvariable=self._search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(toolbar, text="Tag:").pack(side=tk.LEFT, padx=(0, 5))
        self._tag_combo = ttk.Combobox(toolbar, textvariable=self._tag_var, width=15, state='readonly')
        self._tag_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._tag_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh())

        ttk.Label(toolbar, text="Sort:").pack(side=tk.LEFT, padx=(0, 5))
        sort_combo = ttk.Combobox(toolbar, textvariable=self._sort_by, width=10, state='readonly',
                                  values=['updated', 'name', 'size', 'rating', 'ctx'])
        sort_combo.pack(side=tk.LEFT, padx=(0, 10))
        sort_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh())

        ttk.Button(toolbar, text="Scan", command=self._scan).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Refresh", command=self._refresh).pack(side=tk.RIGHT, padx=2)

        # ── Main paned window ──────────────────────────────────────────
        panes = ttk.PanedWindow(self._win, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ── Left panel: model list ─────────────────────────────────────
        left = ttk.Frame(panes, padding=2)
        panes.add(left, weight=1)

        columns = ('name', 'size', 'arch', 'ctx', 'rating')
        self._tree = ttk.Treeview(left, columns=columns, show='tree headings',
                                  selectmode='browse', height=25)
        self._tree.heading('#0', text='')
        self._tree.column('#0', width=0, stretch=False)
        self._tree.heading('name', text='Model')
        self._tree.column('name', width=250, minwidth=150)
        self._tree.heading('size', text='Size')
        self._tree.column('size', width=70, anchor=tk.CENTER)
        self._tree.heading('arch', text='Arch')
        self._tree.column('arch', width=100)
        self._tree.heading('ctx', text='Context')
        self._tree.column('ctx', width=80, anchor=tk.CENTER)
        self._tree.heading('rating', text='★')
        self._tree.column('rating', width=50, anchor=tk.CENTER)

        scroll_y = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll_y.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.bind('<<TreeviewSelect>>', self._on_select)

        # ── Right panel: details ───────────────────────────────────────
        right = ttk.Frame(panes, padding=5)
        panes.add(right, weight=2)

        self._detail_notebook = ttk.Notebook(right)
        self._detail_notebook.pack(fill=tk.BOTH, expand=True)

        # Details tab
        self._details_frame = ttk.Frame(self._detail_notebook, padding=10)
        self._detail_notebook.add(self._details_frame, text='Details')
        self._detail_labels = {}
        for i, field in enumerate(['filename', 'size_gb', 'architecture', 'block_count',
                                    'context_length', 'quantization', 'mmproj', 'path',
                                    'rating', 'last_used', 'created']):
            ttk.Label(self._details_frame, text=f"{field}:", font=('', 9, 'bold')).grid(
                row=i, column=0, sticky=tk.W, pady=1)
            lbl = ttk.Label(self._details_frame, text='', wraplength=400)
            lbl.grid(row=i, column=1, sticky=tk.W, pady=1, padx=(10, 0))
            self._detail_labels[field] = lbl

        # Tags tab
        self._tags_frame = ttk.Frame(self._detail_notebook, padding=10)
        self._detail_notebook.add(self._tags_frame, text='Tags')
        self._tag_checks = {}
        self._tag_refresh_btn = ttk.Button(self._tags_frame, text="Apply Tags", command=self._apply_tags)
        self._tag_refresh_btn.pack(anchor=tk.NE, pady=(0, 10))

        # Training tab
        self._training_frame = ttk.Frame(self._detail_notebook, padding=10)
        self._detail_notebook.add(self._training_frame, text='Training')
        self._training_tree = ttk.Treeview(self._training_frame,
                                           columns=('method', 'status', 'dataset', 'created'),
                                           show='headings', height=8)
        self._training_tree.heading('method', text='Method')
        self._training_tree.heading('status', text='Status')
        self._training_tree.heading('dataset', text='Dataset')
        self._training_tree.heading('created', text='Created')
        self._training_tree.column('method', width=80)
        self._training_tree.column('status', width=80)
        self._training_tree.column('dataset', width=150)
        self._training_tree.column('created', width=140)
        self._training_tree.pack(fill=tk.BOTH, expand=True)

        # Notes tab
        self._notes_frame = ttk.Frame(self._detail_notebook, padding=10)
        self._detail_notebook.add(self._notes_frame, text='Notes')
        self._notes_text = tk.Text(self._notes_frame, height=8, wrap=tk.WORD)
        self._notes_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        ttk.Button(self._notes_frame, text="Save Notes",
                   command=self._save_notes).pack(anchor=tk.E)

        # ── Bottom action bar ──────────────────────────────────────────
        actions = ttk.Frame(self._win, padding=5)
        actions.pack(fill=tk.X)

        self._action_label = ttk.Label(actions, text="Select a model to see details")
        self._action_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(actions, text="Launch Server", command=self._launch).pack(side=tk.RIGHT, padx=2)
        ttk.Button(actions, text="Start Training", command=self._train).pack(side=tk.RIGHT, padx=2)
        ttk.Label(actions, text="Rating:").pack(side=tk.RIGHT, padx=(10, 2))
        self._rating_combo = ttk.Combobox(actions, values=['0', '1', '2', '3', '4', '5'],
                                          width=3, state='readonly')
        self._rating_combo.set('0')
        self._rating_combo.pack(side=tk.RIGHT)
        self._rating_combo.bind('<<ComboboxSelected>>', self._apply_rating)

        # ── Bind search ────────────────────────────────────────────────
        search_entry.bind('<KeyRelease>', lambda e: self._refresh())

        # ── Initial load ───────────────────────────────────────────────
        self._refresh_tags()
        self._refresh()

    def _refresh_tags(self):
        tags = self._repo.list_tags()
        vals = [''] + tags
        self._tag_combo['values'] = vals

    def _refresh(self):
        tag = self._tag_var.get()
        search = self._search_var.get()
        sort = self._sort_by.get()
        models = self._repo.list_models(tag=tag, search=search, sort=sort)

        for item in self._tree.get_children():
            self._tree.delete(item)
        for m in models:
            name = m['filename']
            if m.get('mmproj'):
                name += ' [V]'
            stars = '★' * m.get('rating', 0) + '☆' * (5 - m.get('rating', 0))
            self._tree.insert('', tk.END, iid=str(m['id']),
                              values=(name, f"{m['size_gb']:.1f}G" if m['size_gb'] else '',
                                      m['architecture'], m['context_length'] or '', stars))

        count = len(models)
        self._action_label.config(text=f"{count} model{'s' if count != 1 else ''}")

    def _on_select(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        model_id = int(sel[0])
        model = self._repo.get_model(model_id)
        if not model:
            return
        self._current_model = model
        self._show_details(model)
        self._show_tags(model)
        self._show_training(model)
        self._show_notes(model)
        self._action_label.config(text=f"Selected: {model['filename']}")

    def _show_details(self, m: dict):
        for field, lbl in self._detail_labels.items():
            val = m.get(field, '')
            if field == 'rating':
                val = '★' * int(val) + '☆' * (5 - int(val)) if val else ''
            elif field == 'size_gb':
                val = f"{val:.1f} GB" if val else ''
            elif field == 'mmproj':
                val = val if val else '(none)'
            elif field == 'path':
                # Shorten long paths
                val = val if len(val) < 80 else '...' + val[-77:]
            elif val is None:
                val = ''
            lbl.config(text=str(val))

    def _show_tags(self, m: dict):
        for w in self._tags_frame.winfo_children():
            if isinstance(w, ttk.Checkbutton):
                w.destroy()
        all_tags = self._repo.list_tags()
        model_tags_str = m.get('tags', '') or ''
        model_tags = set(t.strip() for t in model_tags_str.split(',') if t.strip())
        self._tag_checks = {}
        for tag in sorted(all_tags):
            var = tk.BooleanVar(value=tag in model_tags)
            cb = ttk.Checkbutton(self._tags_frame, text=tag, variable=var)
            cb.pack(anchor=tk.W, pady=1)
            self._tag_checks[tag] = var

    def _apply_tags(self):
        if not self._current_model:
            return
        mid = self._current_model['id']
        all_tags = self._repo.list_tags()
        current_tags = set()
        if self._current_model.get('tags'):
            current_tags = set(t.strip() for t in self._current_model['tags'].split(',') if t.strip())

        for tag, var in self._tag_checks.items():
            if var.get() and tag not in current_tags:
                self._repo.add_tag(mid, tag)
            elif not var.get() and tag in current_tags:
                self._repo.remove_tag(mid, tag)
        self._refresh()
        self._refresh_tags()

    def _show_training(self, m: dict):
        for item in self._training_tree.get_children():
            self._training_tree.delete(item)
        runs = self._repo.list_training_runs(base_model_id=m['id'])
        for r in runs:
            self._training_tree.insert('', tk.END, values=(
                r['method'], r['status'], r['dataset'], r['created']))

    def _show_notes(self, m: dict):
        self._notes_text.delete('1.0', tk.END)
        self._notes_text.insert('1.0', m.get('notes', ''))

    def _save_notes(self):
        if not self._current_model:
            return
        notes = self._notes_text.get('1.0', tk.END).strip()
        self._repo.set_notes(self._current_model['id'], notes)
        self._action_label.config(text="Notes saved")

    def _apply_rating(self, event=None):
        if not self._current_model:
            return
        rating = int(self._rating_combo.get())
        self._repo.set_rating(self._current_model['id'], rating)
        self._refresh()
        self._action_label.config(text=f"Rating set to {rating}/5")

    def _scan(self):
        self._action_label.config(text="Scanning models...")
        self._win.update()
        count = self._repo.scan_and_import()
        self._refresh_tags()
        self._refresh()
        self._action_label.config(text=f"Scan complete: {count} models imported/updated")

    def _launch(self):
        if not self._current_model:
            messagebox.showinfo("Launch", "Select a model first")
            return
        result = self._repo.select_and_launch(self._current_model['id'])
        if 'error' in result:
            messagebox.showerror("Launch Error", result['error'])
        else:
            status = "healthy" if result.get('healthy') else "starting"
            msg = (f"Server launched: {result.get('model', '?')}\n"
                   f"Port: {result.get('port', 8080)}\n"
                   f"PID: {result.get('pid', '?')}\n"
                   f"Status: {status}")
            messagebox.showinfo("Launch", msg)

    def _train(self):
        if not self._current_model:
            messagebox.showinfo("Training", "Select a model first")
            return
        # Simple dialog for method selection
        dialog = tk.Toplevel(self._win)
        dialog.title("Start Training")
        dialog.geometry("400x300")
        dialog.transient(self._win)
        dialog.grab_set()

        ttk.Label(dialog, text=f"Model: {self._current_model['filename']}", font=('', 10, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="Training Method:").pack()
        method_var = tk.StringVar(value='sft')
        method_combo = ttk.Combobox(dialog, textvariable=method_var,
                                    values=['sft', 'dpo', 'grpo', 'ppo'], state='readonly')
        method_combo.pack(pady=5)

        ttk.Label(dialog, text="Dataset (name or path):").pack()
        dataset_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=dataset_var, width=50).pack(pady=5)

        ttk.Label(dialog, text="Extra config (JSON):").pack()
        config_text = tk.Text(dialog, height=5, width=50)
        config_text.pack(pady=5)
        config_text.insert('1.0', '{"learning_rate": 2e-5, "num_train_epochs": 3}')

        def do_train():
            cfg = {}
            try:
                raw = config_text.get('1.0', tk.END).strip()
                if raw:
                    cfg = json.loads(raw)
            except json.JSONDecodeError:
                messagebox.showerror("Config Error", "Invalid JSON in config")
                return
            result = self._repo.launch_training(
                self._current_model['id'],
                method=method_var.get(),
                dataset=dataset_var.get().strip(),
                config=cfg,
            )
            dialog.destroy()
            messagebox.showinfo("Training", f"Training run #{result['run_id']} launched")
            self._show_training(self._current_model)

        ttk.Button(dialog, text="Launch Training", command=do_train).pack(pady=20)

    def run(self):
        self._win.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(prog='artifact-picker',
        description='Desktop model picker + artifact repository')
    parser.add_argument('--scan', action='store_true', help='Scan models then open GUI')
    args, _ = parser.parse_known_args()

    repo = ArtifactRepository()
    if args.scan:
        print("Scanning models...")
        n = repo.scan_and_import()
        print(f"  {n} models imported/updated")

    app = ArtifactPicker(repo)
    app.run()


if __name__ == '__main__':
    main()
