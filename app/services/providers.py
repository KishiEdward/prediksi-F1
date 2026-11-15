"""Penyedia state balapan (RaceStateProvider).

Sesuai rencana: predictor TIDAK boleh tahu data berasal dari replay atau live.
- ReplayProvider (Tahap 1): membaca dataset inference dari data/processed.
- LiveProvider   (Tahap 2): stub — live timing FastF1 belum diimplementasikan.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from app.config import get_settings
from app.ml.features import muat_dataset_inference

logger = logging.getLogger(__name__)


@dataclass
class DriverLapState:
    """State satu pembalap pada satu lap (siap diubah menjadi fitur model)."""
    driver_code: str
    driver_id: str
    team_id: str
    lap_number: int
    current_compound: str | None
    tyre_life: int | None


@dataclass
class RaceState:
    """State balapan pada satu lap untuk sekelompok pembalap."""
    year: int
    round: int
    race_name: str
    circuit_id: str
    total_laps: int | None
    lap: int
    track_status: str
    drivers: list[DriverLapState] = field(default_factory=list)


class RaceStateProvider(ABC):
    """Interface sumber data balapan. Predictor hanya bergantung pada ini."""

    @abstractmethod
    def list_seasons(self) -> list[int]:
        """Daftar musim yang tersedia."""

    @abstractmethod
    def list_races(self, year: int) -> list[pd.Series]:
        """Daftar race (baris metadata) untuk satu musim."""

    @abstractmethod
    def list_teams(self, year: int) -> list[dict]:
        """Daftar tim + 2 pembalapnya pada musim tertentu."""

    @abstractmethod
    def get_race_state(self, year: int, round_: int, lap: int,
                       team_id: str | None = None) -> RaceState:
        """Ambil state balapan pada lap tertentu (opsional difilter satu tim)."""

    @abstractmethod
    def get_driver_history(self, year: int, round_: int, driver_code: str,
                           until_lap: int) -> pd.DataFrame:
        """Riwayat baris fitur pembalap dari lap awal s.d. `until_lap` (inklusif).

        Baris HARUS terurut naik per lap dan memuat semua kolom fitur
        (dipakai predictor untuk membangun sliding window).
        """


class ReplayProvider(RaceStateProvider):
    """Tahap 1: replay balapan historis dari dataset hasil ekstraksi."""

    def __init__(self) -> None:
        s = get_settings()
        path_inf = s.processed_dir / "lap_inference.parquet"
        if not path_inf.exists():
            raise FileNotFoundError(
                f"{path_inf} belum ada. Jalankan ekstraksi data + "
                "`python -m app.ml.features` dulu.")
        self.laps: pd.DataFrame = muat_dataset_inference()
        self.races: pd.DataFrame = pd.read_parquet(s.raw_dir / "jolpica_races.parquet")
        self.results: pd.DataFrame = pd.read_parquet(s.raw_dir / "jolpica_results.parquet")
        self.constructors: pd.DataFrame = pd.read_parquet(s.raw_dir / "jolpica_constructors.parquet")
        for df, kolom in [(self.races, "season"), (self.results, "season")]:
            df[kolom] = df[kolom].astype(int)
            df["round"] = df["round"].astype(int)
        logger.info("ReplayProvider siap: %d baris lap, %d race",
                    len(self.laps), len(self.races))

    # ---------- metadata ----------
    def list_seasons(self) -> list[int]:
        return sorted(self.races["season"].unique().tolist())

    def list_races(self, year: int) -> list[pd.Series]:
        r = self.races[self.races["season"] == year].sort_values("round")
        return [row for _, row in r.iterrows()]

    def list_teams(self, year: int) -> list[dict]:
        """Tim musim `year` + maksimal 2 pembalap (yang paling sering tampil)."""
        res = self.results[self.results["season"] == year]
        nama_tim = dict(zip(self.constructors["constructorId"], self.constructors["name"]))
        nama_driver = {}
        if "Driver.givenName" in res.columns:
            for _, row in res.iterrows():
                nama_driver[row["Driver.driverId"]] = (
                    f"{row.get('Driver.givenName', '')} {row.get('Driver.familyName', '')}".strip())

        hasil = []
        for team_id, g in res.groupby("Constructor.constructorId"):
            hitung = (g.groupby("Driver.driverId").size()
                      .sort_values(ascending=False).head(2))
            drivers = [{
                "code": str(g[g["Driver.driverId"] == d]["Driver.code"].iloc[0]),
                "driver_id": str(d),
                "name": nama_driver.get(d, d),
            } for d in hitung.index]
            hasil.append({
                "team_id": str(team_id),
                "team_name": str(nama_tim.get(team_id, team_id)),
                "drivers": drivers,
            })
        return sorted(hasil, key=lambda t: t["team_name"])

    # ---------- state ----------
    def _race_meta(self, year: int, round_: int) -> pd.Series:
        m = self.races[(self.races["season"] == year) & (self.races["round"] == round_)]
        if m.empty:
            raise KeyError(f"Race tidak ditemukan: tahun {year} round {round_}")
        return m.iloc[0]

    def get_race_state(self, year: int, round_: int, lap: int,
                       team_id: str | None = None) -> RaceState:
        meta = self._race_meta(year, round_)
        rows = self.laps[(self.laps["season"] == year) & (self.laps["round"] == round_)
                         & (self.laps["lap_number"] == lap)]
        if rows.empty:
            raise KeyError(f"Tidak ada data lap {lap} untuk race {year} round {round_}")
        if team_id is not None:
            # Tim tanpa data di lap ini -> drivers kosong (bukan error);
            # lapisan API yang memutuskan menjawab 404.
            rows_tim = rows[rows["team_id"] == team_id]
        else:
            rows_tim = rows

        total = rows["total_laps"].iloc[0]
        total = int(total) if pd.notna(total) else None
        status = str(rows["track_status"].iloc[0]) if pd.notna(
            rows["track_status"].iloc[0]) else "GREEN"

        drivers = [
            DriverLapState(
                driver_code=str(r["driver_code"]), driver_id=str(r["driver_id"]),
                team_id=str(r["team_id"]), lap_number=int(r["lap_number"]),
                current_compound=(str(r["current_compound"])
                                  if pd.notna(r["current_compound"]) else None),
                tyre_life=(int(r["tyre_life"]) if pd.notna(r["tyre_life"]) else None),
            ) for _, r in rows_tim.iterrows()
        ]
        return RaceState(
            year=year, round=round_, race_name=str(meta.get("raceName", "")),
            circuit_id=str(meta.get("Circuit.circuitId", "")), total_laps=total,
            lap=lap, track_status=status, drivers=drivers)

    def get_driver_history(self, year: int, round_: int, driver_code: str,
                           until_lap: int) -> pd.DataFrame:
        h = self.laps[(self.laps["season"] == year) & (self.laps["round"] == round_)
                      & (self.laps["driver_code"] == driver_code)
                      & (self.laps["lap_number"] <= until_lap)]
        return h.sort_values("lap_number").reset_index(drop=True)


class LiveProvider(RaceStateProvider):
    """Tahap 2 (OPSIONAL): live timing FastF1 — saat ini masih stub.

    Ketika diimplementasikan, kelas ini membaca sesi berjalan lewat
    `fastf1.get_session(tahun, round, 'R')` + live timing, lalu menghasilkan
    RaceState yang sama bentuknya dengan ReplayProvider. Predictor tidak perlu
    diubah sama sekali.
    """

    def _belum_tersedia(self) -> None:
        raise NotImplementedError(
            "Mode live belum tersedia (Tahap 2). Gunakan mode replay untuk "
            "balapan historis 2021-2025.")

    def list_seasons(self) -> list[int]:
        self._belum_tersedia()

    def list_races(self, year: int) -> list[pd.Series]:
        self._belum_tersedia()

    def list_teams(self, year: int) -> list[dict]:
        self._belum_tersedia()

    def get_race_state(self, year: int, round_: int, lap: int,
                       team_id: str | None = None) -> RaceState:
        self._belum_tersedia()

    def get_driver_history(self, year: int, round_: int, driver_code: str,
                           until_lap: int) -> pd.DataFrame:
        self._belum_tersedia()
