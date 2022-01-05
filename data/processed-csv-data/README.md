# unique.csv


`unique.csv` is the cleaned dataset, produced from the raw data in `../mechanical-turk-data` by the script `../processing-code/python/update_csv.py`.

The cleaning process is detailed [in this issue](https://gitlab.com/SEMERU-Code/Clarity/issues/24). In short, descriptions that are too short (2 words or less) or just say something like "n/a", "nothing to write" are mapped to the empty string to reduce noise in the dataset. As a result, some screenshots end up having all of their descriptions mapped to the empty string. These are excluded from `unique.csv` (if there are no duplicates for it with better, non-empty descriptions). These are also marked as unused in `../master-screen-list/master-list.csv`.

Screenshots that are not present in `../master-screen-list/Clarity-images-SQL.csv` but are present in the manually tagged dataset are also excluded from `unique.csv` (and of course they aren't in `master-list.csv` because `master-list.csv` is derived from `Clarity-images-SQL.csv`).

In the end, `unique.csv` contains **only** screenshots that are present in `Clarity-images-SQL.csv` (filtered screenshots) and screenshots that have at least one substantial description list.


# duplicate.csv

`duplicate.csv` includes duplicate entries in the dataset with less substantial descriptions (based on word count). That is, entries in `unique.csv` have a higher total word count in their descriptions than their duplicates in `duplicate.csv`. Not all entries in `unique.csv` have duplicates, however.

As with `unique.csv`, `duplicate.csv` excludes entries that have all empty descriptions, and it excludes entries that don't appear in `Clarity-images-SQL.csv`.