# python-pwn-productiecapaciteit-infiltratiegebieden
Productiecapaciteit van infiltratiegebieden ICAS en IKIEF

# Install instructions
Via de github desktop applicatie clone deze repository naar een folder op je computer:
```
GitHub desktop applicatie > File > Clone repository... en selecteer bdestombe/python-pwn-productiecapaciteit-infiltratiegebieden en een local path
```

# Werkprotocol
 - Bereid de data voor. See `data/prepare_data.py`
 - Controleer de PT10 hoogteuitlijning: `reports/investigate_pt10_offset4.py` Waarden vastgelegd in `strang_props.csv`: `PA_tag_hleiding`.
 - Bereken de weerstandsmodellen
   - Leidingweerstand: `reports/Leidingweerstand.py`
   - Putfilterweerstand: `reports/putfilter3.py`
   - Weerstand watervoerendpakket: `reports/wvpweerstand.py`
 - Analyseer de modeluitkomsten in de betreffende mappen in de `results` map
   - Controleer of de data compleet is
   - Controleer of de werkzaamheden effectief waren
   - Controleer of de modelfit redelijk is
 - Synthese `reports/losses_rapport.py`. Analyseer het effect van een schoonmaak op de productiecapaciteit van een strang.
   - Pas de datum voor de geplande schoonmaak aan
   - Run de code
   - Resultaten staan in `results/Synthese`
 - Synthese `reports/somrapport.py`. Analyseer de som van de productiecapaciteit van verschillende/alle strangen.