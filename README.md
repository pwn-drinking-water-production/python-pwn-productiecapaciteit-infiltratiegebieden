# python-pwn-productiecapaciteit-infiltratiegebieden
Productiecapaciteit van infiltratiegebieden ICAS en IKIEF

# Install instructions
Via de github desktop applicatie clone deze repository naar een folder op je computer:

- https://github.com/pwn-drinking-water-production/python-pwn-productiecapaciteit-infiltratiegebieden
- C:\PythonScripts\Repositories\pwn-drinking-water-production\python-pwn-productiecapaciteit-infiltratiegebieden

Navigeer naar de gecloonde repository en maak het environment aan met:
```
uv venv --python=3.14
```
Activeer het environment en installeer de `productiecapaciteit`package met:
```
.venv\Scripts\activate
uv pip install -e .
```
Je Python kernel staat nu in `.venv\Scripts\python.exe`


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