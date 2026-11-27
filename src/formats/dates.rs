//! Dates as formats keep them (Julian days, GPS seconds, a year and its day), written in ISO 8601.

/// The proleptic Gregorian year, month and day `days` after 1970-01-01.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    // In 1..=31 and 1..=12.
    (y, u32::try_from(m).unwrap_or(1), u32::try_from(d).unwrap_or(1))
}

/// `secs` after 1970-01-01T00:00:00 as `YYYY-MM-DDThh:mm:ss`, to the nearest second, with a `Z` when
/// `utc` (else it's a local time, its zone unknown). `None` outside the years 1 to 9999.
pub(crate) fn unix(secs: f64, utc: bool) -> Option<String> {
    if !secs.is_finite() || secs.abs() > 3e11 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, reason = "checked to be in range above")]
    let s = secs.round() as i64;
    let (y, mo, d) = civil(s.div_euclid(86_400));
    let t = s.rem_euclid(86_400);
    if !(1..=9999).contains(&y) {
        return None;
    }
    let z = if utc { "Z" } else { "" };
    Some(format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}{z}",
        t / 3600,
        t / 60 % 60,
        t % 60
    ))
}

/// A Julian date (days since noon, 1 January 4713 BC), as [`unix`] writes it.
#[cfg_attr(not(feature = "dxf"), allow(dead_code, reason = "not every reader uses it"))]
pub(crate) fn julian(jd: f64, utc: bool) -> Option<String> {
    unix((jd - 2_440_587.5) * 86_400.0, utc)
}

/// GPS time (seconds since 1980-01-06T00:00:00 UTC, without leap seconds) as a UTC [`unix`] time.
#[cfg_attr(not(feature = "e57"), allow(dead_code, reason = "not every reader uses it"))]
pub(crate) fn gps(secs: f64) -> Option<String> {
    // When GPS time got each second ahead of UTC, in GPS seconds.
    const LEAPS: [f64; 18] = [
        46_828_800.0,
        78_364_801.0,
        109_900_802.0,
        173_059_203.0,
        252_028_804.0,
        315_187_205.0,
        346_723_206.0,
        393_984_007.0,
        425_520_008.0,
        457_056_009.0,
        504_489_610.0,
        551_750_411.0,
        599_184_012.0,
        820_108_813.0,
        914_803_214.0,
        1_025_136_015.0,
        1_119_744_016.0,
        1_167_264_017.0,
    ];
    let behind = LEAPS.iter().filter(|&&t| secs >= t).count();
    #[allow(clippy::cast_precision_loss, reason = "at most 18")]
    let behind = behind as f64;
    unix(secs + 315_964_800.0 - behind, true)
}

/// Day `day` (1 is 1 January) of `year` as `YYYY-MM-DD`. `None` for a day the year doesn't have.
#[cfg_attr(not(feature = "las"), allow(dead_code, reason = "not every reader uses it"))]
pub(crate) fn day_of_year(year: u16, day: u16) -> Option<String> {
    let y = i64::from(year);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    if !(1..=9999).contains(&y) || day == 0 || day > 365 + u16::from(leap) {
        return None;
    }
    // Days from 1970-01-01 to 1 January of `year`.
    let p = y - 1;
    let jan1 = 365 * p + p / 4 - p / 100 + p / 400 - 719_162;
    let (y, m, d) = civil(jan1 + i64::from(day) - 1);
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_in_iso_8601() {
        assert_eq!(unix(0.0, true).as_deref(), Some("1970-01-01T00:00:00Z"));
        assert_eq!(
            unix(951_782_400.0 + 3661.0, false).as_deref(),
            Some("2000-02-29T01:01:01")
        );
        assert_eq!(julian(2_451_545.0, true).as_deref(), Some("2000-01-01T12:00:00Z"));
        assert_eq!(julian(f64::NAN, true), None);
        assert_eq!(julian(0.0, true), None, "before the year 1");
        // GPS time ran 18 seconds ahead of UTC from 2017.
        assert_eq!(gps(0.0).as_deref(), Some("1980-01-06T00:00:00Z"));
        assert_eq!(gps(1_300_000_018.0).as_deref(), Some("2021-03-17T07:06:40Z"));
        assert_eq!(day_of_year(2024, 60).as_deref(), Some("2024-02-29"));
        assert_eq!(day_of_year(2023, 365).as_deref(), Some("2023-12-31"));
        assert_eq!(day_of_year(2023, 366), None);
        assert_eq!(day_of_year(2024, 0), None);
        assert_eq!(day_of_year(1970, 1).as_deref(), Some("1970-01-01"));
    }
}
