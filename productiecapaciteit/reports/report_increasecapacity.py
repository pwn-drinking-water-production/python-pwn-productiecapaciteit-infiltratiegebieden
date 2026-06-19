"""
Report to show how to increase capacity of the system
Start with IKIEF 
Goals: 
1) how many m3/h does each "strang" currently produce at a suction pressure of -7.5 mwk?
2) Per strang per element: how much m3/h can the production be increased by improving that element (e.g. the well clogging)
    Elements:
    1) infiltration lake clogging
    2) borehole wall/ nearby aquifer / annulus clogging
    3) screen clogging
    4) "strangleiding" diameter (fix clogging amount @ 1yr since last maintenance)
    5) "verzamelleiding" diameter (fix clogging amount @ 1yr since last maintenance)

# Actions
0) Validate pressure sensors (PTofsett)
1) Get WVPweerstand (infiltration lake + well clogging) 
    - Validate method of computation
    - Validate relationship with Q
2) Get filterweerstand (screen resistance)
    - Validate method of computation
    - Validate relationship with Q
3) Get leidingweerstand (pipe resistance)
    - Validate method of computation
    - Validate relationship with Q
    - fit to D'arcy-Weisbach? 
    - Split into pipe resistance north-south and pipe resistance of the pipe towards the secundair
4) Create graph m3/h (x) vs required suction pressure (y) for CURRENT situation
    - With bedrijfsvoering: get realistic max suction pressure
    - plot max suction pressure => get max m3/h
5) Create scenario's for:
    - pandschoonmaak
    - chemische regeneratie (HD is regular maintenance, not part of this report)
    - redrill wells / extra wells
    - increase diameter pipe N-S
    - increase diameter pipe towards secundair
6) create graphs / obtain estimate of increase in max m3/h per strang
7) sit with Amon to determine how many m3/h he should take into account (scenario's?)
"""


