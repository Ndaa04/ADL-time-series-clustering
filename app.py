"""
🌾 Analisis Fenologi Padi — NDVI Clustering Pipeline
Alur vertikal. Canvas sampling tanpa copy-paste.
"""
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from data_loader import load_raw_data, apply_smoothing
from clustering import run_clustering, get_dtw_description
from visualization import (
    plot_grid_preview,
    plot_smoothing_preview,
    plot_sample_grid,
    plot_sample_ts_preview,
    calculate_metrics,
    plot_comparison,
    plot_individual_clusters,
    plot_spatial_map,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG & CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Indeks Vegetasi", page_icon="🌾",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .step-card {
    border: 1.5px solid #dcedc8; border-left: 5px solid #388e3c;
    border-radius: 10px; padding: 1.4rem 1.8rem 1rem;
    margin-bottom: 0.5rem; background: #f9fff9;
  }
  .step-card h3 { margin-top: 0; color: #1b5e20; font-size: 1.1rem; }
  .step-divider { border: none; border-top: 2px dashed #c8e6c9; margin: 1.8rem 0; }
  .badge-ok   { display:inline-block; background:#e8f5e9; color:#2e7d32;
                border:1px solid #a5d6a7; border-radius:20px;
                padding:.2rem .9rem; font-size:.85rem; font-weight:600; }
  .badge-warn { display:inline-block; background:#fff8e1; color:#e65100;
                border:1px solid #ffe082; border-radius:20px;
                padding:.2rem .9rem; font-size:.85rem; font-weight:600; }
  /* bridge input: tampilkan tapi readonly style */
  [data-testid="stTextInput"] input { 
    font-size: 11px !important; 
    color: #2e7d32 !important;
    background: #f1f8e9 !important;
    border: 1px solid #a5d6a7 !important;
    cursor: default;
  }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1b5e20,#388e3c,#1b5e20);
            padding:1.8rem 2.5rem;border-radius:12px;margin-bottom:2rem;color:white;">
  <h1 style="margin:0;font-size:1.9rem;">🌾 Analisis Indeks Vegetasi</h1>
  <p style="margin:.3rem 0 0;opacity:.85;">Pipeline NDVI · DTW · HDBSCAN — Kabupaten Lamongan</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA  (+  peta grid langsung tampil)
# ═══════════════════════════════════════════════════════
st.markdown('<div class="step-card"><h3>📁 Step 1 — Load Data</h3>', unsafe_allow_html=True)

if "df_raw" not in st.session_state:
    if st.button("⬇️ Muat Data", type="primary", use_container_width=True):
        bar = st.progress(0)
        try:
            df_raw, nr, nc = load_raw_data(
                progress_callback=lambda p, t: bar.progress(p, text=t))
            bar.empty()
            st.session_state.update(df_raw=df_raw, nr=nr, nc=nc)
            st.rerun()
        except Exception as e:
            bar.empty()
            st.error(f"❌ Gagal membaca data: {e}")
            st.stop()
else:
    df_raw = st.session_state["df_raw"]
    nr, nc = st.session_state["nr"], st.session_state["nc"]
    n_lok = df_raw['id_lokasi'].nunique()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lokasi Unik",  f"{n_lok:,}")
    m2.metric("Total Baris",  f"{len(df_raw):,}")
    m3.metric("Dimensi Grid", f"{nr}×{nc}")
    m4.metric("Periode", f"{df_raw['tanggal'].min().date()} → {df_raw['tanggal'].max().date()}")
    st.markdown('<span class="badge-ok">✅ Data siap</span>', unsafe_allow_html=True)

    # Peta grid semua data — tampil sebelum pilih tahun
    with st.expander("🗺️ Peta Grid Seluruh Lokasi", expanded=True):
        st.pyplot(plot_grid_preview(df_raw, nr, nc,
                                    title='Sebaran Seluruh Lokasi (2023 & 2024)'))

st.markdown('</div>', unsafe_allow_html=True)
if "df_raw" not in st.session_state:
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 2 — PILIH TAHUN
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📅 Step 2 — Pilih Tahun Analisis</h3>', unsafe_allow_html=True)

c1, c2, _ = st.columns([1, 1, 5])
for yr, col in [("2023", c1), ("2024", c2)]:
    is_active = st.session_state.get("tahun") == yr
    if col.button(f"📆 {yr}",
                  type="primary" if is_active else "secondary",
                  use_container_width=True):
        if st.session_state.get("tahun") != yr:
            for k in ["df_smooth", "df_year", "sampled_ids", "pivot_df", "_bridge_ids"]:
                st.session_state.pop(k, None)
        st.session_state["tahun"] = yr
        st.rerun()

if "tahun" in st.session_state:
    st.markdown(f'<span class="badge-ok">✅ Tahun: {st.session_state["tahun"]}</span>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if "tahun" not in st.session_state:
    st.info("👆 Pilih tahun terlebih dahulu.")
    st.stop()

tahun     = st.session_state["tahun"]
df_raw_yr = df_raw[df_raw["tahun"] == tahun].copy()


# ═══════════════════════════════════════════════════════
#  STEP 3 — PREPROCESSING (Savitzky-Golay)
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>⚙️ Step 3 — Preprocessing (Savitzky-Golay)</h3>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
window_size = c1.slider("Window Size",      5, 61, 31, 2, help="Harus ganjil")
poly_order  = c2.slider("Polynomial Order", 1,  5,  2)

if st.button("▶ Jalankan Smoothing", type="primary", use_container_width=True):
    bar = st.progress(0)
    df_smooth = apply_smoothing(df_raw_yr, window_size, poly_order,
                                progress_callback=lambda p, t: bar.progress(p, text=t))
    bar.empty()
    st.session_state["df_smooth"] = df_smooth
    st.session_state["df_year"]   = df_smooth[df_smooth["tahun"] == tahun].copy()
    for k in ["sampled_ids", "pivot_df", "_bridge_ids"]:
        st.session_state.pop(k, None)
    st.rerun()

if "df_smooth" in st.session_state:
    df_year = st.session_state["df_year"]
    n_ts = df_year["tanggal"].nunique()
    st.markdown(
        f'<span class="badge-ok">✅ {df_year["id_lokasi"].nunique():,} lokasi × {n_ts} hari</span>',
        unsafe_allow_html=True)
    # Preview NDVI asli vs smooth
    with st.expander("📈 Preview NDVI Asli vs Smoothed (3 lokasi acak)", expanded=True):
        st.pyplot(plot_smoothing_preview(df_year, n=3))

st.markdown('</div>', unsafe_allow_html=True)

if "df_smooth" not in st.session_state:
    st.info("👆 Klik **Jalankan Smoothing** untuk melanjutkan.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 4 — SAMPLING (Canvas Paint Interaktif)
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>🎯 Step 4 — Sampling Data (Paint Mode)</h3>',
            unsafe_allow_html=True)

df_year = st.session_state["df_year"]

id_to_pos = (df_year[['id_lokasi', 'grid_row', 'grid_col']]
             .drop_duplicates('id_lokasi').copy())

grid_cells    = [{"r": int(r.grid_row), "c": int(r.grid_col), "id": r.id_lokasi}
                 for r in id_to_pos.itertuples()]
grid_json     = json.dumps(grid_cells)
nr_js, nc_js  = int(nr), int(nc)
init_selected = json.dumps(st.session_state.get("sampled_ids", []))

CANVAS_HTML = f"""
<!DOCTYPE html><html>
<head>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:sans-serif;background:#f9fff9;padding:6px;}}
  #toolbar{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
            margin-bottom:6px;padding:7px 10px;
            background:#fff;border:1px solid #c8e6c9;border-radius:8px;}}
  #toolbar label{{font-size:13px;color:#333;}}
  #counter{{font-weight:700;color:#2e7d32;font-size:14px;margin-left:auto;
            background:#e8f5e9;padding:3px 12px;border-radius:20px;}}
  button{{padding:5px 14px;border:none;border-radius:6px;
          font-size:13px;cursor:pointer;font-weight:600;}}
  #btn-random {{background:#1976d2;color:#fff;}}
  #btn-clear  {{background:#e53935;color:#fff;}}
  #btn-confirm{{background:#388e3c;color:#fff;}}
  button:hover{{opacity:.85;}}
  #canvas-outer{{display:flex;justify-content:center;margin-top:6px;}}
  #canvas-wrap{{overflow:auto;border:1px solid #c8e6c9;border-radius:8px;background:#fff;display:inline-block;}}
  canvas{{display:block;cursor:crosshair;}}
  #msg{{margin-top:6px;font-size:12px;text-align:center;min-height:16px;color:#555;}}
</style>
</head>
<body>
<div id="toolbar">
  <label>Brush:
    <input id="brush" type="range" min="1" max="14" value="3"
           style="width:80px;vertical-align:middle;">
    <span id="brush-val">3</span>
  </label>
  <span style="font-size:11px;color:#888;margin-left:2px;">🖱 kiri=pilih · kanan=hapus</span>
  <label style="margin-left:4px;">Random:
    <input id="n-random" type="number" value="100" min="1"
           style="width:70px;padding:3px 5px;border:1px solid #ccc;border-radius:4px;">
  </label>
  <button id="btn-random">🎲 Random</button>
  <button id="btn-clear" >🗑 Clear</button>
  <button id="btn-confirm">✅ Konfirmasi Seleksi</button>
  <span id="counter">Terseleksi: 0</span>
</div>

<div id="canvas-outer">
  <div id="canvas-wrap"><canvas id="c"></canvas></div>
</div>
<div id="msg"></div>

<script>
const NR={nr_js}, NC={nc_js};
const GRID_CELLS={grid_json};
const INIT_SEL={init_selected};

const C_EMPTY='#D3D3D3', C_AVAIL='rgba(80,200,80,0.35)', C_SEL='#2196F3';

const available=new Uint8Array(NR*NC);
const selected =new Uint8Array(NR*NC);
const cellId   =new Array(NR*NC).fill(null);

GRID_CELLS.forEach(d=>{{
  const i=d.r*NC+d.c;
  available[i]=1; cellId[i]=d.id;
}});
const initSet=new Set(INIT_SEL);
GRID_CELLS.forEach(d=>{{ if(initSet.has(d.id)) selected[d.r*NC+d.c]=1; }});

// --- Canvas size: fit lebar parent frame ---
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const WRAP=document.getElementById('canvas-wrap');

// Canvas mengisi ~82% lebar layar, proporsional tinggi
const availW = Math.min(window.innerWidth * 0.82, 1200);
const availH = window.innerHeight * 0.72;
const CELL = Math.max(4, Math.floor(Math.min(availW / NC, availH / NR)));
canvas.width  = NC * CELL;
canvas.height = NR * CELL;

function draw(){{
  for(let r=0;r<NR;r++){{
    for(let c=0;c<NC;c++){{
      const i=r*NC+c;
      ctx.fillStyle = !available[i] ? C_EMPTY : selected[i] ? C_SEL : C_AVAIL;
      ctx.fillRect(c*CELL, r*CELL, CELL-1, CELL-1);
    }}
  }}
  const n=selected.reduce((s,v)=>s+v,0);
  document.getElementById('counter').textContent='Terseleksi: '+n.toLocaleString();
}}
draw();

// Brush slider
const brushSlider=document.getElementById('brush');
document.getElementById('brush-val').textContent=brushSlider.value;
brushSlider.addEventListener('input',()=>document.getElementById('brush-val').textContent=brushSlider.value);

function applyBrush(x,y,mode){{
  const col0=Math.floor(x/CELL), row0=Math.floor(y/CELL), R=+brushSlider.value;
  for(let r=Math.max(0,row0-R);r<=Math.min(NR-1,row0+R);r++){{
    for(let c=Math.max(0,col0-R);c<=Math.min(NC-1,col0+R);c++){{
      if(Math.sqrt((r-row0)**2+(c-col0)**2)<=R){{
        const i=r*NC+c; if(available[i]) selected[i]=mode;
      }}
    }}
  }}
}}

let dragging=false, dragMode=0;
canvas.addEventListener('mousedown',e=>{{
  e.preventDefault(); dragging=true; dragMode=e.button===2?0:1;
  const rc=canvas.getBoundingClientRect();
  applyBrush(e.clientX-rc.left,e.clientY-rc.top,dragMode); draw();
}});
canvas.addEventListener('mousemove',e=>{{
  if(!dragging) return;
  const rc=canvas.getBoundingClientRect();
  applyBrush(e.clientX-rc.left,e.clientY-rc.top,dragMode); draw();
}});
window.addEventListener('mouseup',()=>{{dragging=false;}});
canvas.addEventListener('contextmenu',e=>e.preventDefault());

// Touch
canvas.addEventListener('touchstart',e=>{{
  e.preventDefault(); dragging=true; dragMode=1;
  const t=e.touches[0],rc=canvas.getBoundingClientRect();
  applyBrush(t.clientX-rc.left,t.clientY-rc.top,1); draw();
}},{{passive:false}});
canvas.addEventListener('touchmove',e=>{{
  e.preventDefault();
  const t=e.touches[0],rc=canvas.getBoundingClientRect();
  applyBrush(t.clientX-rc.left,t.clientY-rc.top,dragMode); draw();
}},{{passive:false}});
canvas.addEventListener('touchend',()=>{{dragging=false;}});

// Random
document.getElementById('btn-random').addEventListener('click',()=>{{
  selected.fill(0);
  const avail=[]; GRID_CELLS.forEach(d=>avail.push(d.r*NC+d.c));
  const n=Math.min(+document.getElementById('n-random').value||100, avail.length);
  for(let i=avail.length-1;i>0;i--){{
    const j=Math.floor(Math.random()*(i+1));
    [avail[i],avail[j]]=[avail[j],avail[i]];
  }}
  for(let i=0;i<n;i++) selected[avail[i]]=1;
  draw();
}});

// Clear
document.getElementById('btn-clear').addEventListener('click',()=>{{
  selected.fill(0); draw();
}});

// Konfirmasi → inject ke hidden Streamlit text_input lalu trigger rerun
document.getElementById('btn-confirm').addEventListener('click',()=>{{
  const ids=[];
  for(let r=0;r<NR;r++) for(let c=0;c<NC;c++){{
    const i=r*NC+c; if(selected[i]&&cellId[i]) ids.push(cellId[i]);
  }}
  const msg=document.getElementById('msg');
  if(!ids.length){{ msg.textContent='⚠️ Belum ada lokasi dipilih!'; msg.style.color='#e53935'; return; }}

  const payload=JSON.stringify(ids);
  // Cari label dengan teks "lokasi_terpilih", ambil input di wrapper yang sama
  let found=false;
  const labels=window.parent.document.querySelectorAll('label');
  labels.forEach(lbl=>{{
    if(lbl.textContent.trim()==='lokasi_terpilih'){{
      const wrapper=lbl.closest('[data-testid="stTextInput"]');
      if(wrapper){{
        const inp=wrapper.querySelector('input');
        if(inp){{
          const setter=Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set;
          setter.call(inp, payload);
          inp.dispatchEvent(new Event('input',{{bubbles:true}}));
          found=true;
        }}
      }}
    }}
  }});

  msg.textContent = found
    ? '✅ '+ids.length+' lokasi dikonfirmasi — scroll ke bawah untuk lanjut.'
    : '⚠️ Bridge tidak ditemukan. Coba refresh halaman.';
  msg.style.color = found ? '#2e7d32' : '#e53935';
}});
</script>
</body></html>
"""

components.html(CANVAS_HTML, height=680, scrolling=False)

# ── Bridge input: letaknya DI BAWAH canvas, disembunyikan via CSS ────
bridge_val = st.text_input("lokasi_terpilih", value="", key="canvas_bridge",
                            placeholder="ID lokasi akan muncul di sini setelah konfirmasi...")

# ── Proses nilai bridge ketika berubah ───────────────────────
raw_bridge = st.session_state.get("canvas_bridge", "").strip()

if raw_bridge and raw_bridge != st.session_state.get("_bridge_ids", ""):
    try:
        ids_list = json.loads(raw_bridge)
        if isinstance(ids_list, list) and len(ids_list) > 0:
            st.session_state["sampled_ids"] = ids_list
            st.session_state["_bridge_ids"] = raw_bridge
            st.session_state.pop("pivot_df", None)
            st.session_state.pop("dist_matrix", None)
            st.session_state.pop("show_sample_result", None)
            st.rerun()
    except json.JSONDecodeError:
        pass

# ── Tampilkan hasil sampling jika sudah ada ───────────────────
if "sampled_ids" in st.session_state:
    sampled_ids = st.session_state["sampled_ids"]
    st.markdown(
        f'<span class="badge-ok">✅ {len(sampled_ids):,} lokasi terkonfirmasi</span>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶ Proses Sampel", type="primary", use_container_width=True,
                 key="btn_proses_sampel"):
        st.session_state["show_sample_result"] = True
        st.rerun()

if st.session_state.get("show_sample_result") and "sampled_ids" in st.session_state:
    sampled_ids = st.session_state["sampled_ids"]
    tab1, tab2 = st.tabs(["🗺️ Peta Sebaran Sampel", "📈 Preview Time Series"])
    with tab1:
        st.pyplot(plot_sample_grid(df_year, sampled_ids, nr, nc))
    with tab2:
        st.pyplot(plot_sample_ts_preview(df_year, sampled_ids, n=3))

st.markdown('</div>', unsafe_allow_html=True)

if "sampled_ids" not in st.session_state:
    st.info("👆 Pilih lokasi di canvas lalu klik **✅ Konfirmasi Seleksi**.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 5 — DTW
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📊 Step 5 — Hitung DTW</h3>', unsafe_allow_html=True)

sampled_ids = st.session_state["sampled_ids"]
df_sampled  = df_year[df_year["id_lokasi"].isin(sampled_ids)].copy()
dtw_info    = get_dtw_description(df_sampled)

cols = st.columns(len(dtw_info))
for col, (k, v) in zip(cols, dtw_info.items()):
    col.metric(k, v)

with st.expander("Lihat sampel data (5 baris)"):
    st.dataframe(df_sampled[['id_lokasi', 'tanggal', 'NDVI', 'NDVI_smooth',
                              'lat_y', 'lon_x']].head(5), use_container_width=True)

if st.button("▶ Hitung DTW", type="primary", use_container_width=True):
    bar = st.progress(0)
    # Hanya DTW, simpan dist_matrix ke session
    from tslearn.metrics import cdist_dtw
    pivot_raw = df_sampled.pivot(index='id_lokasi', columns='tanggal', values='NDVI_smooth')
    pivot_raw = pivot_raw.ffill(axis=1).bfill(axis=1)
    bar.progress(0.1, "📐 Pivoting...")
    data_3d = pivot_raw.values[:, :, np.newaxis].astype(np.float32)
    bar.progress(0.2, f"⏳ Menghitung DTW ({len(pivot_raw)} lokasi)...")
    dist_matrix = cdist_dtw(data_3d, n_jobs=-1, verbose=0)
    bar.progress(1.0, "✅ DTW selesai!")
    bar.empty()
    st.session_state["pivot_raw"]   = pivot_raw
    st.session_state["dist_matrix"] = dist_matrix
    st.session_state.pop("pivot_df", None)
    st.success(f"✅ DTW selesai — matriks {len(pivot_raw)}×{len(pivot_raw)}")
    st.rerun()

if "dist_matrix" in st.session_state:
    dm = st.session_state["dist_matrix"]
    st.markdown(
        f'<span class="badge-ok">✅ Matriks DTW siap: {dm.shape[0]}×{dm.shape[1]}</span>',
        unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if "dist_matrix" not in st.session_state:
    st.info("👆 Klik **Hitung DTW** untuk melanjutkan.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 6 — CLUSTERING HDBSCAN
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>🔍 Step 6 — Clustering HDBSCAN</h3>', unsafe_allow_html=True)
st.caption("Atur parameter lalu klik Jalankan. Bisa diulang berkali‑kali tanpa hitung DTW ulang.")

c1, c2, c3 = st.columns(3)
min_cluster_size = c1.slider("Min Cluster Size", 2, 20, 3)
min_samples      = c2.slider("Min Samples",      1, 10, 2)
epsilon          = c3.slider("Epsilon",           0.0, 0.5, 0.05, 0.01)

if st.button("🚀 Jalankan HDBSCAN", type="primary", use_container_width=True):
    import hdbscan as hdbscan_lib
    bar = st.progress(0, text="🔍 Menjalankan HDBSCAN...")
    dist_matrix  = st.session_state["dist_matrix"]
    pivot_raw    = st.session_state["pivot_raw"]

    clusterer = hdbscan_lib.HDBSCAN(
        metric='precomputed',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        gen_min_span_tree=True,
        prediction_data=True
    )
    labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
    bar.progress(1.0, "✅ Selesai!")
    bar.empty()

    pivot_df = pivot_raw.copy()
    pivot_df['cluster'] = labels
    st.session_state["pivot_df"] = pivot_df

    n_cls   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    st.success(f"✅ {n_cls} cluster terbentuk · {n_noise} noise")
    st.rerun()

if "pivot_df" in st.session_state:
    pv = st.session_state["pivot_df"]
    n_cls   = len(set(pv['cluster'])) - (1 if -1 in pv['cluster'].values else 0)
    n_noise = int((pv['cluster'] == -1).sum())
    st.markdown(
        f'<span class="badge-ok">✅ {n_cls} cluster · {n_noise} noise</span>',
        unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

if "pivot_df" not in st.session_state:
    st.info("👆 Klik **Jalankan HDBSCAN** untuk memulai.")
    st.stop()


# ═══════════════════════════════════════════════════════
#  STEP 7 — HASIL AKHIR
# ═══════════════════════════════════════════════════════
st.markdown('<hr class="step-divider">', unsafe_allow_html=True)
st.markdown('<div class="step-card"><h3>📈 Step 7 — Hasil Akhir</h3>', unsafe_allow_html=True)

pivot_df = st.session_state["pivot_df"]
df_fenologi, cluster_ts, valid_statuses, df_labeled = calculate_metrics(df_year, pivot_df)

if df_fenologi.empty:
    st.warning("⚠️ Tidak ada cluster valid. Coba turunkan Min Cluster Size atau sesuaikan Epsilon.")
else:
    st.subheader(f"📰 Tabel Metrik Fenologi ({tahun})")
    st.dataframe(
        df_fenologi.style
        .format({'Puncak NDVI':'{:.3f}','Min NDVI':'{:.3f}','Amplitudo':'{:.3f}',
                 'Rata-rata NDVI':'{:.3f}','Rata-rata StdDev':'{:.3f}','Jml Titik':'{:,.0f}'})
        .background_gradient(cmap='YlGn', subset=['Puncak NDVI','Amplitudo']),
        use_container_width=True)

    st.subheader("📊 Grafik Time Series")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_comparison(cluster_ts, valid_statuses, tahun))
    with col2:
        st.pyplot(plot_individual_clusters(cluster_ts, valid_statuses))

    st.subheader("🗺️ Peta Sebaran Cluster")
    st.pyplot(plot_spatial_map(df_labeled, nr, nc, valid_statuses,
                               sampled_ids=st.session_state.get("sampled_ids")))

st.markdown('</div>', unsafe_allow_html=True)
