# Air Traffic Big Data Project

The objective of the project is to design and implement an aviation data analytics system that enables the identification and evaluation of factors with the greatest impact on flight delays and cancellations. The project leverages Big Data technologies to process large-scale datasets and to present analytical results in the form of clear, aggregated metrics and visualizations.

## Data
### Aviation Data

#### Data provided by BTS (Bureau of Transportation Statistics)

1. **Airline On-Time Performance Data**
   
   The dataset contains information on commercial flights operated within the United States. It includes details such as the date of the flight, airline identifier, aircraft identifier, origin and destination airports, as well as scheduled and actual departure and arrival times. Additionally, the dataset provides information on whether a flight was cancelled or diverted, along with the reasons for delays.

   - Data format: `.csv`
   - Update frequency: monthly (data available since 1987)
   - Approximate number of records per month: 540,000

3. **Aviation Support Tables: Master Coordinate Tables**
   
   This dataset contains more detailed information about airports, including their geographic location (coordinates, state/region), airport name, served market (city), local time zone, year of establishment, and operational status (open/closed).

   - Data format: `.csv`
   - Update frequency: irregular (no fixed publication interval)
   - Approximate number of records: 20,000

#### Data provided by FAA (Federal Aviation Administration)

3. **Aircraft Registry**
   
   The dataset contains information on aircraft registered in the United States. Due to regulations in force in the U.S., commercial flights may be operated only by aircraft registered with the FAA. Therefore, this registry provides comprehensive coverage of aircraft relevant to the project.

   - Data format: `.txt`
   - Update frequency: daily
   - Approximate number of records: 309,000


### Weather Data

4. **Storm Events**
   
   Data related to extreme weather events are provided by **NCEI (National Centers for Environmental Information)**. The dataset includes information about the type of weather event, its classification category, severity, start and end time, and location of occurrence. Some weather phenomena, such as tornadoes or hurricanes, may span large geographic areas and distances. The dataset also includes information on fatalities, injuries, and estimated property damage.

   - Data format: `.csv`
   - Update frequency: annual (data available since 1950)
   - Approximate number of records per year: 63,000
