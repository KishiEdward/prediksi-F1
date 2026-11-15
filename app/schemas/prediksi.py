"""Skema Pydantic v2 untuk request/response API.

Catatan framing: seluruh keluaran adalah PREDIKSI/PERKIRAAN keputusan pit
observasional (pola keputusan tim pada data historis), BUKAN rekomendasi
strategi optimal.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompoundBerikutnya(BaseModel):
    """Prediksi compound pada pit stop berikutnya + probabilitas tiap kelas."""
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Compound dengan probabilitas tertinggi")
    probabilities: dict[str, float] = Field(
        description="Probabilitas tiap compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET)")


class JendelaPit(BaseModel):
    """Interval pit window terkalibrasi split-conformal dari residual CV.

    BUKAN P10-P90 mentah quantile regression (coverage hanya 52.8%).
    Interval simetris [pred - q, pred + q] dengan q dikalibrasi agar
    coverage empiris = tingkat keyakinan nominal di fold cv2 (val 2024, bersih).
    """
    model_config = ConfigDict(extra="forbid")

    lap_bawah: int = Field(ge=0, description="Batas bawah perkiraan lap pit berikutnya")
    lap_atas: int = Field(ge=0, description="Batas atas perkiraan lap pit berikutnya")
    tingkat_keyakinan: float = Field(
        default=0.8, ge=0, le=1,
        description="Coverage nominal interval (0.8 = 80%)")
    coverage_kalibrasi: float = Field(
        ge=0, le=1,
        description="Coverage aktual terukur di set kalibrasi cv2 (val 2024)")
    setengah_lebar: float = Field(
        ge=0, description="q = kuantil |residual| yang dipakai membentuk interval")


class PeringatanPrediksi(BaseModel):
    """Caveat akurasi kontekstual untuk satu prediksi pembalap."""
    model_config = ConfigDict(extra="forbid")

    compound_basah_tidak_andal: bool = Field(
        default=False,
        description="True jika compound prediksi INTERMEDIATE/WET — akurasi sangat rendah "
                    "(test 2025: INTER F1=0.005 sup=94, WET F1=0 sup=0)")
    bucket_jarak: str | None = Field(
        default=None,
        description="Bucket jarak ke pit berdasarkan prediksi: '0-5', '6-15', '>15'")
    mae_bucket_test: float | None = Field(
        default=None,
        description="MAE historis test 2025 untuk bucket jarak ini")
    catatan: list[str] = Field(
        default_factory=list,
        description="Catatan keterbatasan tambahan yang relevan untuk prediksi ini")


class PrediksiPembalap(BaseModel):
    """Prediksi untuk satu pembalap pada lap tertentu."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Kode 3 huruf pembalap, contoh VER")
    driver_id: str
    current_compound: str | None = Field(description="Compound yang sedang dipakai")
    tyre_life: int | None = Field(description="Umur ban saat ini (lap)")
    predicted_laps_to_pit: int = Field(
        ge=0, description="Perkiraan sisa lap menuju pit stop berikutnya")
    predicted_pit_lap: int | None = Field(
        description="Perkiraan nomor lap pit stop berikutnya")
    next_compound: CompoundBerikutnya
    probabilitas_masih_pit: float | None = Field(
        default=None, ge=0, le=1,
        description="Probabilitas pembalap masih akan pit lagi sebelum finis "
                    "(binary head terpisah, DNF di-flag sensor bukan dilabel 0). "
                    "None jika model biner tidak dimuat.")
    jendela_pit: JendelaPit | None = Field(
        default=None,
        description="Interval pit window terkalibrasi split-conformal dari residual CV. "
                    "Bukan P10-P90 mentah quantile regression.")
    peringatan: PeringatanPrediksi | None = Field(
        default=None,
        description="Caveat akurasi kontekstual: keandalan compound basah, "
                    "MAE per bucket jarak ke pit, catatan keterbatasan.")


class ResponsReplay(BaseModel):
    """Response endpoint /predict/replay."""
    model_config = ConfigDict(extra="forbid")

    year: int
    round: int
    lap: int
    team: str
    race_name: str
    track_status: str = Field(description="Status lintasan pada lap tersebut (GREEN/SC/VSC/RED)")
    drivers: list[PrediksiPembalap]
    note: str = ("Prediksi observasional berdasarkan pola keputusan tim "
                 "pada data 2021-2025, bukan strategi optimal.")
    model_version: str


class RaceMeta(BaseModel):
    """Metadata satu seri balapan."""
    model_config = ConfigDict(extra="forbid")

    year: int
    round: int
    race_name: str
    circuit_id: str
    circuit_name: str
    date: str
    total_laps: int | None = None


class DriverMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    driver_id: str
    name: str


class TeamMeta(BaseModel):
    """Tim beserta dua pembalapnya pada musim tersebut."""
    model_config = ConfigDict(extra="forbid")

    team_id: str
    team_name: str
    drivers: list[DriverMeta] = Field(min_length=0, max_length=2)


class ModelInfo(BaseModel):
    """Informasi model yang sedang dimuat API."""
    model_config = ConfigDict(extra="forbid")

    model_version: str
    trained_at: str | None = None
    metrics: dict | None = Field(default=None, description="Ringkasan metrik evaluasi")
    artifact_suffix: str | None = Field(
        default=None,
        description="Suffix artefak per-seed (Run 1/2). Kosong/null untuk Run 3 xgb_separate.")
    window_lap: int = Field(
        description="Panjang sliding window LSTM. Untuk Run 3 (xgb_separate) bernilai 1 "
                    "karena prediksi per-baris tanpa sequence.")
    compounds: list[str]
    split: dict
    primary_model: str | None = Field(
        default=None,
        description="Tipe model aktif: 'xgb_separate' (Run 3), 'lstm', atau 'hybrid' (Run 1/2)")
    conformal: dict | None = Field(
        default=None,
        description="Parameter kalibrasi split-conformal pit window (alpha, q, coverage)")
    binary_head: dict | None = Field(
        default=None,
        description="Metrik head biner 'masih pit?' (accuracy, f1, roc_auc, data_stats)")
    caveats: dict | None = Field(
        default=None,
        description="Caveat global: wet_unreliable, accuracy_by_distance, conformal_window")
    cv_selection: dict | None = Field(
        default=None,
        description="Pemenang CV per target + catatan bias cv1/cv2")


class ResponsError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str


class ResponsLiveBelumTersedia(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str = ("Prediksi live belum tersedia (Tahap 2). Saat ini API hanya "
                   "mendukung mode replay balapan historis 2021-2025.")
