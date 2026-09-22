# Melbourne Heat and Road Accidents

Does hot weather predict more road accidents in Melbourne's south east, once the time of year and a longer run trend are already accounted for? This started as a university assignment trying to answer that question and ended up with a genuinely different answer than the one it originally reported, after fixing two real problems in how it built its data.

The full analysis is in [`analysis.md`](analysis.md) (also viewable as [`analysis.html`](analysis.html)), generated from the R Markdown source in [`analysis.Rmd`](analysis.Rmd). It's a companion to the [Victorian Car Accident Severity Analysis](../victorian-car-accident-analysis/) project, using the same accident dataset with a different region and a different statistical question.

## What was wrong with the original

- **The accident total for the target region was built from the wrong columns.** The original picked out "Metropolitan South East" accidents with a column range that was hard coded, `daily_accidents[c(9, 10, 11, 12)]`. That range actually pulls in one column from a different region (Metropolitan North West) and drops one of South East's own four severity columns. The outcome variable being modeled was never quite the region it was labeled as.
- **The "Excess Heat Factor" wasn't the Excess Heat Factor.** The original computed today's maximum temperature minus the average of the previous three days, and called that EHF. The actual Excess Heat Factor, the metric the Bureau of Meteorology uses to rate heatwave severity, compares the last three days against both a threshold for extreme temperature and the typical temperature over the past month, so a heatwave has to be both hot and a genuine jump from what's normal for that time of year. This version implements the real formula.
- **Weather and accident data were joined by assuming matching row order, not by date.** Referencing one data frame's column directly inside a model formula built on a different data frame only works if both are sorted identically and have exactly the same rows. Here they're joined explicitly on date instead.

## Method

1. Rebuild the Metropolitan South East daily accident total from named columns instead of a range that was hard coded.
2. Load Moorabbin Airport weather station data and compute a properly defined Excess Heat Factor.
3. Join accidents and weather by date.
4. Compare a linear trend, a seasonal smooth, and a model combining trend with season on equal footing, rather than comparing two models answering different questions.
5. Add heat and rainfall to the best model from step 4, and check whether they're actually significant once trend and season are already accounted for.

## Findings

**Accidents in this region genuinely declined over the five years in the data, independent of season.** The trend coefficient is negative and highly significant (p < 2e-16) in every model that includes it.

**There's a real seasonal pattern**, peaking around the middle of February and again sharply from the middle to the end of November, both consistent with holiday travel periods.

![Seasonal smooth of accidents by day of year](figures/plot-seasonal-1.png)

**Heat looked like it mattered, until the trend was properly accounted for.** Adding the Excess Heat Factor to a season-only model does lower the AIC. But that comparison was missing the trend term, and hot days aren't spread evenly across the five years, so EHF's smooth term was partly standing in for the missing trend rather than capturing a real heat effect. In the full model, with trend, season, heat, and rain all included together, EHF is not significant (p = 0.18) and neither is rainfall (p = 0.73).

The honest answer to the original question: once you know the time of year and account for the fact that accidents were trending down across the five years, knowing how hot or wet a given day was doesn't add anything further. That's a less exciting finding than "heatwaves cause accidents," but it's the one the data actually supports, and it's a good example of why a trend term has to be checked before treating a weather effect as real.

## Repo structure

```
melbourne-heat-and-accidents/
├── analysis.Rmd     # R Markdown source, runs the full analysis
├── analysis.md       # knitted output with all figures inline
├── analysis.html     # same, as a standalone HTML file
├── data/
│   ├── car_accidents_victoria.csv
│   └── moorabbin_weather.csv
└── figures/          # individual chart PNGs
```

## Running it

Requires R with `tidyverse`, `mgcv`, `broom`, and `RcppRoll` installed.

```r
rmarkdown::render("analysis.Rmd")
```

## Limitations

Five years of station data is a short baseline for the 95th percentile threshold the Excess Heat Factor needs; the Bureau's own calculations use a climatology spanning several decades, so "extreme" here means extreme for this short record, not the standardised figure over the long run. The downward trend is also not explained here, just detected. It could reflect genuine road safety improvements, changes in reporting, or something else entirely, and would need data outside this dataset to pin down.
