# Data Pipeline Module

## Overview
This module implements a raw-to-relational pipeline:
- Scrape ≥ 3 categories from [books.toscrape.com](http://books.toscrape.com)
- Clean and enrich data
- Load into normalized SQLite schema
- Query with SQL and pandas

## Install
```bash
pip install -r requirements.txt