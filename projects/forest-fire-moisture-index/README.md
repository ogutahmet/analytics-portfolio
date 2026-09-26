# Predicting Fire Moisture Code from Weather Aggregation Functions

The Fine Fuel Moisture Code (FFMC) is part of the Canadian Forest Fire Weather Index system: a reading, taken on the day itself, of how dry a forest's litter layer is, which drives how easily a fire starts and spreads. This project fits several weighted aggregation functions, weighted arithmetic and power means, an ordered weighted average (OWA), and a Choquet integral, to predict FFMC from a handful of weather readings taken that same day, then checks that against plain linear regression.

The full analysis is in [`analysis.md`](analysis.md) (also viewable as [`analysis.html`](analysis.html)), generated from the R Markdown source in [`analysis.Rmd`](analysis.Rmd).

## Dataset

517 fire records from the northeast region of Portugal, a cut of the well known Montesinho forest fire dataset limited to numerical variables (Cortez & Morais, 2007), with FFMC, three other fire weather indices, temperature, humidity, wind, rain, and burned area. This started as a university assignment on the same dataset. The original report is a useful reference for what to try, but a couple of things in it don't hold up on a closer look, both explained below.

## What changed from the original coursework version

- **One of the four chosen predictors ran the wrong direction in time.** The original picked temperature, humidity, wind, and burned area to predict FFMC. Burned area is only known once a fire has already happened, so it can't sensibly predict a fire weather reading taken on the day itself, and its correlation with FFMC here is close to zero anyway. This version predicts FFMC from temperature, humidity, wind, and rain instead: the same four weather readings the real FFMC formula is actually built from.
- **The original's own results table doesn't match its own variable selection.** Its report picks four variables and builds an array with five columns to fit on, but the fitted model's weight table lists eight weights, not four, meaning the model that produced those results wasn't actually built from the four chosen variables.
- **Humidity was fed into the aggregation functions the wrong way round.** Weighted means, OWA, and the Choquet integral can only combine inputs that move in the same direction as the outcome. Humidity has a real negative relationship with FFMC (drier air, higher FFMC), so it needs to be flipped, using "dryness" instead of "humidity", before it can help. This one change alone cut every aggregation function's error by 25-30%.

## Method

1. Sample 300 of the 517 records with a fixed seed, and explore each variable's relationship with FFMC.
2. Select temperature, humidity, wind, and rain (the actual meteorological inputs to the real FFMC calculation), scale them to [0, 1] by their minimum and maximum, and complement any variable with a negative correlation to FFMC so every input moves in a consistent direction.
3. Fit five aggregation functions with the provided library: a weighted arithmetic mean, power means with p = 0.5 and p = 2, an OWA, and a Choquet integral.
4. Predict FFMC for a case set aside from fitting, and compare it to the measured value.
5. Fit a linear regression on the same four variables and compare its error against the best aggregation function.

## Findings

**Every model built from a weighted mean collapsed onto a single variable rather than a genuine blend.** The weighted arithmetic and power mean models all put 100% of their weight on one variable (relative humidity, once corrected for direction) and 0% on the rest; OWA and the Choquet integral both converged to simply taking the largest of the four scaled readings each day. This isn't a coding error, it's a real property of how these functions are fit: with four weakly and inconsistently correlated predictors, the linear program that finds the weights can land on a rule using just one variable rather than a blend.

![Error by aggregation function](figures/plot-comparison-1.png)

**OWA and the Choquet integral outperform the models built on a mean, but linear regression beats all of them by a wide margin.** The weighted and power means land around 26 FFMC units of error (on a scale spanning about 77.5); OWA and Choquet do better, around 18. Linear regression's error is about 5.9, roughly a third of the best aggregation function's. That gap has a structural explanation: weighted means, OWA, and the Choquet integral can only output a value between the smallest and largest of that day's inputs, and FFMC in this dataset is heavily concentrated near the top of its range. Linear regression isn't bound by its inputs' range; its intercept alone sits close to that ceiling, leaving each variable to only nudge the prediction up or down from there.

![Linear regression vs. the best aggregation function](figures/lm-vs-best-1.png)

**The prediction on the case set aside lands reasonably close, with a caveat.** Both the best aggregation function and the linear model land in a broadly plausible range for the test case given in the brief, but two of the four input readings sit right at or beyond the edge of the training sample's observed range, so the model is extrapolating a little rather than interpolating within data it has actually seen.

## Repo structure

```
forest-fire-moisture-index/
├── analysis.Rmd       # R Markdown source, runs the full analysis
├── analysis.md          # knitted output with all figures inline
├── analysis.html        # same, as a standalone HTML file
├── AggWaFit718.R        # provided library implementing the aggregation functions
├── data/
│   ├── fires.txt         # the 517-row dataset
│   ├── transformed.txt   # the 300-row scaled sample used for fitting
│   ├── out_*.txt         # per-model predictions
│   └── stats_*.txt       # per-model error measures and fitted weights
└── figures/              # individual chart PNGs
```

## Running it

Requires R with `lpSolve` installed.

```r
rmarkdown::render("analysis.Rmd")
```

## Limitations

The sample used for fitting is 300 of 517 available records; a different random sample would shift the exact weights, though the broad pattern (models built on a mean collapsing to one variable, OWA and Choquet outperforming them, linear regression outperforming everything) held up consistently across the checks done here. The Choquet integral is fit with full generality (`kadd` equal to the number of variables), which is tractable for four inputs but wouldn't scale cleanly to a much larger variable set.
