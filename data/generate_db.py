"""Generate synthetic travel database CSVs.

Run: python data/generate_db.py
Overwrites all four CSVs in data/travel_db/.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)

OUT = Path(__file__).parent / "travel_db"

CITIES = [
  "New York",
  "Los Angeles",
  "Chicago",
  "Miami",
  "San Francisco",
  "Las Vegas",
  "Seattle",
  "Boston",
  "New Orleans",
  "Denver",
  "Austin",
  "Nashville",
]

AIRLINES = ["Delta", "United", "American", "Southwest", "JetBlue", "Alaska"]

# city -> list of hotel names
HOTEL_NAMES: dict[str, list[str]] = {
  "New York": [
    "The Manhattan Grand", "Brooklyn Loft Hotel", "Central Park Suites",
    "Hudson Yards Inn", "Times Square Premier", "Soho Boutique Hotel",
    "The Midtown Classic", "East River Lodge", "Fifth Avenue Hotel", "NoMad House",
  ],
  "Los Angeles": [
    "Sunset Strip Hotel", "Beverly Hills Retreat", "Santa Monica Pier Inn",
    "Hollywood Grand", "The Silver Lake", "Downtown LA Loft",
    "Venice Beach Resort", "Griffith Park Lodge", "Echo Park Hotel", "Culver City Inn",
  ],
  "Chicago": [
    "The Magnificent Mile Hotel", "Millennium Park Lodge", "Wicker Park Inn",
    "Navy Pier Suites", "The Loop Hotel", "River North Grand",
    "Lincoln Park Retreat", "Hyde Park Inn", "Gold Coast Suites", "Pilsen Boutique",
  ],
  "Miami": [
    "South Beach Grand", "Art Deco Inn", "Brickell City Hotel",
    "Wynwood Boutique", "Coconut Grove Resort", "Little Havana Suites",
    "Coral Gables Hotel", "Key Biscayne Lodge", "Design District Inn", "Midtown Miami Hotel",
  ],
  "San Francisco": [
    "Union Square Hotel", "Fisherman's Wharf Inn", "Castro Boutique",
    "The Embarcadero Grand", "Mission District Loft", "Nob Hill Suites",
    "Chinatown Hotel", "Pacific Heights Lodge", "SoMa Hotel", "The Marina",
  ],
  "Las Vegas": [
    "The Strip Grand", "Fremont Boutique", "Desert Palms Resort",
    "Neon Lights Hotel", "The High Roller Inn", "Arts District Suites",
    "Paradise Valley Lodge", "Summerlin Hotel", "Henderson Resort", "Downtown Vegas",
  ],
  "Seattle": [
    "Pike Place Inn", "Capitol Hill Hotel", "The Space Needle Lodge",
    "Queen Anne Suites", "Ballard Boutique", "Fremont Hotel",
    "Pioneer Square Grand", "South Lake Union Inn", "Belltown Hotel", "Columbia City Lodge",
  ],
  "Boston": [
    "Beacon Hill Inn", "Back Bay Grand", "The Fenway Hotel",
    "North End Suites", "South Boston Lodge", "Cambridge Boutique",
    "Charlestown Harbor Hotel", "Jamaica Plain Inn", "Brookline Suites", "The Seaport",
  ],
  "New Orleans": [
    "French Quarter Grand", "Garden District Inn", "The Marigny Hotel",
    "Warehouse District Suites", "Bywater Boutique", "Uptown Lodge",
    "Mid-City Hotel", "Tremé Inn", "Lakeview Suites", "Algiers Point Hotel",
  ],
  "Denver": [
    "Mile High Hotel", "LoDo Boutique", "Capitol Hill Inn",
    "Cherry Creek Grand", "RiNo Art Hotel", "Washington Park Lodge",
    "Baker District Suites", "Highland Hotel", "Stapleton Inn", "Globeville Loft",
  ],
  "Austin": [
    "Sixth Street Grand", "South Congress Inn", "The Domain Hotel",
    "East Austin Boutique", "Rainey Street Suites", "Zilker Lodge",
    "Mueller Hotel", "Hyde Park Inn", "North Loop Suites", "Travis Heights Hotel",
  ],
  "Nashville": [
    "Broadway Grand", "The Gulch Hotel", "12South Inn",
    "Music Row Suites", "Germantown Boutique", "East Nashville Lodge",
    "Sylvan Park Hotel", "Midtown Suites", "The Nations Inn", "Belmont Hotel",
  ],
}

AMENITIES_POOL = [
  "WiFi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Parking",
  "Pet-friendly", "Rooftop", "Concierge", "Laundry", "Business center",
]

# city -> list of (name, cuisine)
RESTAURANTS: dict[str, list[tuple[str, str]]] = {
  "New York": [
    ("Katz's Deli", "American"), ("Nobu NYC", "Japanese"), ("Le Bernardin", "French"),
    ("Grimaldi's", "Italian"), ("Xi'an Famous Foods", "Chinese"), ("Shake Shack", "American"),
    ("Peter Luger Steak", "American"), ("Ivan Ramen", "Japanese"), ("Taïm", "Mediterranean"),
    ("Juliana's Pizza", "Italian"), ("Balthazar", "French"), ("Momofuku Noodle Bar", "Asian"),
    ("Keens Steakhouse", "American"), ("Sylvia's Restaurant", "Soul Food"),
    ("Di Fara Pizza", "Italian"), ("Marea", "Italian"), ("Sushi Nakazawa", "Japanese"),
    ("The Halal Guys", "Middle Eastern"), ("Russ & Daughters", "Jewish Deli"),
    ("Sammy's Roumanian", "Romanian"),
  ],
  "Los Angeles": [
    ("In-N-Out Burger", "American"), ("Nobu Malibu", "Japanese"), ("Republique", "French"),
    ("Osteria Mozza", "Italian"), ("Cassia", "Southeast Asian"), ("Gjusta", "Bakery"),
    ("Bestia", "Italian"), ("Guelaguetza", "Mexican"), ("Night + Market", "Thai"),
    ("Perle", "American"), ("Dialogue", "American"), ("Taco Maria", "Mexican"),
    ("Pizzeria Mozza", "Italian"), ("Howlin' Ray's", "Southern"), ("Yxta Cocina", "Mexican"),
    ("Bavel", "Middle Eastern"), ("Kismet", "Mediterranean"), ("Sqirl", "Californian"),
    ("Jon & Vinny's", "Italian"), ("Sushi Zo", "Japanese"),
  ],
  "Chicago": [
    ("Alinea", "American"), ("Girl & The Goat", "American"), ("Lou Malnati's", "Pizza"),
    ("Portillo's", "American"), ("Au Cheval", "American"), ("Smyth", "American"),
    ("Avec", "Mediterranean"), ("Big Jones", "Southern"), ("Publican", "American"),
    ("Lula Cafe", "American"), ("Monteverde", "Italian"), ("The Purple Pig", "Mediterranean"),
    ("Frontera Grill", "Mexican"), ("Japonais", "Japanese"), ("RPM Italian", "Italian"),
    ("Kasama", "Filipino"), ("Duck Duck Goat", "Chinese"), ("Topolobampo", "Mexican"),
    ("Chicago Chop House", "American"), ("MingHin Cuisine", "Chinese"),
  ],
  "Miami": [
    ("Joe's Stone Crab", "Seafood"), ("Versailles Restaurant", "Cuban"),
    ("KYU Miami", "Asian BBQ"), ("Zuma Miami", "Japanese"), ("Coya Miami", "Peruvian"),
    ("Los Fuegos", "Argentine"), ("Michael's Genuine", "American"), ("Mandolin Aegean", "Greek"),
    ("Stubborn Seed", "American"), ("Fooq's", "American"), ("Chotto Matte", "Japanese/Peruvian"),
    ("La Mar", "Peruvian"), ("Coyo Taco", "Mexican"), ("Pubbelly Sushi", "Japanese"),
    ("Phuc Yea", "Vietnamese"), ("The Surf Club Restaurant", "Seafood"),
    ("Swan", "American"), ("Cervantes Signature", "Cuban"), ("Ariete", "American"),
    ("DIRT", "Vegetarian"),
  ],
  "San Francisco": [
    ("Tartine Manufactory", "Bakery"), ("Zuni Cafe", "American"), ("Lazy Bear", "American"),
    ("State Bird Provisions", "American"), ("Rich Table", "American"), ("Nopa", "American"),
    ("Burma Superstar", "Burmese"), ("Cotogna", "Italian"), ("Quince", "Italian"),
    ("Benu", "Korean/American"), ("Atelier Crenn", "French"), ("The Progress", "American"),
    ("Liholiho Yacht Club", "Hawaiian"), ("Sorrel", "American"), ("Saison", "American"),
    ("Dosa", "Indian"), ("Aziza", "Moroccan"), ("Kin Khao", "Thai"),
    ("Z & Y Restaurant", "Chinese"), ("Flour + Water", "Italian"),
  ],
  "Las Vegas": [
    ("Joel Robuchon", "French"), ("Joël Robuchon Restaurant", "French"),
    ("é by José Andrés", "Spanish"), ("CUT by Wolfgang Puck", "Steakhouse"),
    ("Momofuku", "Asian"), ("Carbone", "Italian"), ("Nobu Las Vegas", "Japanese"),
    ("Bouchon", "French"), ("Herb N Kitchen", "American"), ("Wicked Spoon", "Buffet"),
    ("Yardbird", "Southern"), ("Gordon Ramsay Hell's Kitchen", "American"),
    ("Secret Pizza", "Pizza"), ("In-N-Out Burger", "American"), ("Rao's Las Vegas", "Italian"),
    ("Bavette's", "Steakhouse"), ("Osteria Fiorella", "Italian"), ("Lotus of Siam", "Thai"),
    ("Esther's Kitchen", "Italian"), ("Carson Kitchen", "American"),
  ],
  "Seattle": [
    ("Canlis", "American"), ("Nue", "International"), ("Bateau", "Steakhouse"),
    ("The Walrus and the Carpenter", "Seafood"), ("Altura", "Italian"),
    ("How to Cook a Wolf", "Italian"), ("Din Tai Fung", "Chinese"), ("Spinasse", "Italian"),
    ("Sitka & Spruce", "American"), ("Sushi Kashiba", "Japanese"),
    ("Taylor Shellfish Farms", "Seafood"), ("Le Pichet", "French"),
    ("Poppy", "American"), ("Taneda", "Japanese"), ("Harvest Vine", "Spanish"),
    ("Marjorie", "American"), ("Toulouse Petit", "Cajun"), ("Matt's in the Market", "Seafood"),
    ("Wild Ginger", "Asian"), ("Serious Pie", "Pizza"),
  ],
  "Boston": [
    ("Neptune Oyster", "Seafood"), ("Oleana", "Mediterranean"), ("Menton", "French/Italian"),
    ("Island Creek Oyster Bar", "Seafood"), ("Myers + Chang", "Asian"),
    ("Tasting Counter", "American"), ("Ostra", "Seafood"), ("Row 34", "Seafood"),
    ("Tiger Mama", "Southeast Asian"), ("Giulia", "Italian"),
    ("The Salty Pig", "American"), ("Toro", "Spanish"), ("Coppa", "Italian"),
    ("Waypoint", "American"), ("Central Kitchen", "American"), ("Sarma", "Middle Eastern"),
    ("Sweet Cheeks Q", "BBQ"), ("Clio", "French"), ("Deuxave", "French"),
    ("Harvest", "American"),
  ],
  "New Orleans": [
    ("Commander's Palace", "Creole"), ("Dooky Chase's", "Soul Food/Creole"),
    ("Galatoire's", "French Creole"), ("Antoine's", "French Creole"),
    ("Cochon", "Cajun"), ("August", "American"), ("Cafe Du Monde", "Cafe"),
    ("Acme Oyster House", "Seafood"), ("Peche Seafood Grill", "Seafood"),
    ("Coquette", "American"), ("Compère Lapin", "Caribbean"), ("MoPho", "Vietnamese"),
    ("Shaya", "Israeli"), ("Willa Jean", "American"), ("Costera", "Mexican/American"),
    ("Johnny's Po-Boys", "Cajun"), ("Domilise's Po-Boys", "Cajun"),
    ("Turkey and the Wolf", "American"), ("High Hat Cafe", "American"),
    ("Lilette", "French"),
  ],
  "Denver": [
    ("Rioja", "Mediterranean"), ("Acorn", "American"), ("Guard and Grace", "Steakhouse"),
    ("Work & Class", "American"), ("Mercantile", "American"), ("Beast + Bottle", "American"),
    ("Sushi Den", "Japanese"), ("Potager", "American"), ("Root Down", "American"),
    ("Leven Deli", "Jewish Deli"), ("Mizuna", "American"), ("Jax Fish House", "Seafood"),
    ("Fruition", "American"), ("El Five", "Mediterranean"), ("Tavernetta", "Italian"),
    ("Linger", "International"), ("ViewHouse", "American"), ("Hop Alley", "Chinese"),
    ("Williams & Graham", "American"), ("Brider", "American"),
  ],
  "Austin": [
    ("Franklin Barbecue", "BBQ"), ("Uchi", "Japanese"), ("Odd Duck", "American"),
    ("Sour Duck Market", "American"), ("La Barbecue", "BBQ"), ("Launderette", "American"),
    ("Lenoir", "American"), ("Ramen Tatsu-ya", "Japanese"), ("Barley Swine", "American"),
    ("1886 Cafe & Bakery", "American"), ("Qui", "American"), ("Justine's Brasserie", "French"),
    ("Uchiko", "Japanese"), ("Salt & Time", "American"), ("Emmer & Rye", "American"),
    ("Juniper", "American"), ("Loro", "Asian BBQ"), ("Vera Cruz All Natural", "Mexican"),
    ("Valentina's Tex Mex BBQ", "Tex-Mex"), ("Fresa's", "Mexican"),
  ],
  "Nashville": [
    ("Prince's Hot Chicken", "Southern"), ("The Catbird Seat", "American"),
    ("Rolf and Daughters", "Italian"), ("Husk Nashville", "Southern"),
    ("Arnold's Country Kitchen", "Southern"), ("City House", "Italian"),
    ("The Pharmacy Burger", "American"), ("Mas Tacos", "Mexican"),
    ("Josephine", "American"), ("Sinema", "American"), ("Bastion", "American"),
    ("Chauhan Ale & Masala House", "Indian"), ("The Southern Steak", "American"),
    ("Margot Cafe", "French/Italian"), ("Henrietta Red", "Seafood"),
    ("Milk and Honey", "American"), ("Butcher & Bee", "American"),
    ("Etch", "American"), ("Dino's", "American"), ("Attaboy", "American"),
  ],
}

# city -> list of (name, category, base_price, duration_hours)
ACTIVITIES: dict[str, list[tuple[str, str, float, float]]] = {
  "New York": [
    ("Central Park Bike Tour", "Outdoor", 35, 2.5),
    ("Metropolitan Museum of Art", "Culture", 25, 3.0),
    ("Statue of Liberty Ferry", "Sightseeing", 24, 3.5),
    ("Broadway Show", "Entertainment", 120, 2.5),
    ("High Line Walk", "Outdoor", 0, 1.5),
    ("Brooklyn Bridge Walk", "Outdoor", 0, 1.0),
    ("One World Observatory", "Sightseeing", 43, 1.5),
    ("NY Food Tour – Greenwich Village", "Food", 65, 3.0),
    ("MoMA Visit", "Culture", 30, 2.5),
    ("Yankee Stadium Tour", "Sports", 25, 1.5),
    ("Chelsea Market Visit", "Food", 0, 2.0),
    ("Helicopter Tour", "Sightseeing", 175, 1.0),
    ("Jazz at Blue Note", "Entertainment", 40, 2.0),
    ("Harlem Gospel Tour", "Culture", 45, 2.5),
    ("Kayaking in Hudson River", "Outdoor", 30, 2.0),
  ],
  "Los Angeles": [
    ("Universal Studios Hollywood", "Entertainment", 110, 8.0),
    ("Getty Museum", "Culture", 0, 3.0),
    ("Hollywood Sign Hike", "Outdoor", 0, 3.5),
    ("Venice Beach Bike Ride", "Outdoor", 20, 2.0),
    ("Griffith Observatory", "Culture", 0, 2.0),
    ("Warner Bros. Studio Tour", "Entertainment", 75, 3.0),
    ("Santa Monica Pier", "Sightseeing", 0, 2.0),
    ("LACMA Visit", "Culture", 25, 3.0),
    ("Malibu Wine Safari", "Food", 120, 4.0),
    ("Runyon Canyon Hike", "Outdoor", 0, 2.0),
    ("Sunset Strip Walking Tour", "Sightseeing", 25, 2.0),
    ("Kayaking in Marina del Rey", "Outdoor", 40, 2.0),
    ("LA Food Tour", "Food", 75, 3.5),
    ("The Broad Museum", "Culture", 0, 2.5),
    ("Surfing Lesson in Santa Monica", "Sports", 90, 2.0),
  ],
  "Chicago": [
    ("Art Institute of Chicago", "Culture", 25, 3.0),
    ("Chicago Architecture Boat Tour", "Sightseeing", 55, 1.5),
    ("Millennium Park Visit", "Outdoor", 0, 1.5),
    ("Navy Pier", "Sightseeing", 0, 2.0),
    ("Willis Tower Skydeck", "Sightseeing", 30, 1.0),
    ("Chicago Deep Dish Pizza Tour", "Food", 65, 3.0),
    ("Wrigley Field Tour", "Sports", 25, 1.5),
    ("Lincoln Park Zoo", "Outdoor", 0, 2.5),
    ("Chicago Blues Bar Crawl", "Entertainment", 40, 3.0),
    ("Museum of Science and Industry", "Culture", 22, 3.0),
    ("360 Chicago Observatory", "Sightseeing", 28, 1.0),
    ("Chicago Jazz Walk", "Entertainment", 35, 2.5),
    ("Kayaking on the Chicago River", "Outdoor", 45, 2.0),
    ("Second City Comedy Show", "Entertainment", 55, 2.0),
    ("Field Museum", "Culture", 24, 3.0),
  ],
  "Miami": [
    ("South Beach Walking Tour", "Sightseeing", 0, 2.0),
    ("Everglades Airboat Tour", "Outdoor", 65, 3.5),
    ("Wynwood Walls Tour", "Culture", 0, 1.5),
    ("Snorkeling at Biscayne Bay", "Outdoor", 55, 3.0),
    ("Miami Beach Bike Rental", "Outdoor", 20, 2.0),
    ("Pérez Art Museum Miami", "Culture", 16, 2.0),
    ("Little Havana Food Tour", "Food", 60, 2.5),
    ("Sunset Sailing Cruise", "Sightseeing", 55, 2.0),
    ("Dolphin Tour", "Outdoor", 65, 2.5),
    ("Vizcaya Museum & Gardens", "Culture", 25, 2.0),
    ("Miami Nightlife Tour", "Entertainment", 50, 3.0),
    ("Paddleboarding in Biscayne Bay", "Sports", 40, 2.0),
    ("Art Basel Visit", "Culture", 0, 4.0),
    ("Key West Day Trip", "Sightseeing", 85, 8.0),
    ("Miami Food & Cocktail Tour", "Food", 75, 3.0),
  ],
  "San Francisco": [
    ("Alcatraz Island Tour", "Culture", 45, 3.0),
    ("Golden Gate Bridge Walk", "Outdoor", 0, 2.0),
    ("Fisherman's Wharf Visit", "Sightseeing", 0, 2.0),
    ("Cable Car Ride", "Sightseeing", 8, 0.5),
    ("Muir Woods Day Trip", "Outdoor", 15, 4.0),
    ("SF Food Tour – Mission District", "Food", 70, 3.0),
    ("Palace of Fine Arts", "Culture", 0, 1.5),
    ("Bike the Golden Gate Bridge", "Outdoor", 45, 3.0),
    ("Chinatown Walking Tour", "Culture", 25, 2.0),
    ("SFMOMA", "Culture", 25, 3.0),
    ("Wine Country Day Trip (Napa)", "Food", 130, 8.0),
    ("Exploratorium", "Culture", 30, 3.0),
    ("Bay Cruise", "Sightseeing", 35, 1.5),
    ("Castro District Tour", "Culture", 25, 2.0),
    ("Kayaking in the Bay", "Outdoor", 65, 2.5),
  ],
  "Las Vegas": [
    ("Grand Canyon Helicopter Tour", "Sightseeing", 350, 4.0),
    ("Hoover Dam Tour", "Sightseeing", 45, 4.0),
    ("Cirque du Soleil O Show", "Entertainment", 120, 2.0),
    ("High Roller Observation Wheel", "Sightseeing", 35, 1.0),
    ("Mob Museum", "Culture", 30, 2.0),
    ("Zion National Park Day Trip", "Outdoor", 110, 8.0),
    ("Las Vegas Food Tour", "Food", 65, 3.0),
    ("Neon Museum", "Culture", 25, 1.5),
    ("Vegas Strip Night Tour", "Sightseeing", 30, 2.0),
    ("Flyboard Experience", "Sports", 100, 1.0),
    ("Magic Show", "Entertainment", 65, 2.0),
    ("Off-Road ATV Adventure", "Outdoor", 130, 3.0),
    ("Las Vegas Pool Party", "Entertainment", 40, 4.0),
    ("Comedy Club", "Entertainment", 55, 2.0),
    ("Antelope Canyon Tour", "Outdoor", 150, 8.0),
  ],
  "Seattle": [
    ("Space Needle", "Sightseeing", 38, 1.5),
    ("Pike Place Market Tour", "Food", 35, 2.0),
    ("Chihuly Garden and Glass", "Culture", 32, 1.5),
    ("Mt. Rainier Day Trip", "Outdoor", 85, 8.0),
    ("Underground Tour", "Culture", 22, 1.5),
    ("Seattle Art Museum", "Culture", 20, 2.5),
    ("Kayaking on Lake Union", "Outdoor", 55, 2.5),
    ("Museum of Pop Culture", "Culture", 30, 2.0),
    ("Ferry to Bainbridge Island", "Sightseeing", 9, 3.0),
    ("Coffee Crawl", "Food", 40, 2.5),
    ("Olympic National Park Day Trip", "Outdoor", 95, 8.0),
    ("Burke Museum", "Culture", 15, 2.0),
    ("Seattle Food Tour", "Food", 70, 3.0),
    ("Whale Watching Tour", "Outdoor", 110, 4.0),
    ("Amazon Spheres Visit", "Culture", 0, 1.5),
  ],
  "Boston": [
    ("Freedom Trail Walking Tour", "Culture", 0, 3.0),
    ("Duck Boat Tour", "Sightseeing", 45, 1.5),
    ("Museum of Fine Arts", "Culture", 27, 3.0),
    ("Boston Harbor Cruise", "Sightseeing", 35, 1.5),
    ("Harvard & MIT Campus Walk", "Culture", 0, 3.0),
    ("New England Aquarium", "Culture", 32, 2.0),
    ("Fenway Park Tour", "Sports", 25, 1.5),
    ("Boston Food Tour", "Food", 70, 3.0),
    ("Salem Witch Trials Tour", "Culture", 30, 4.0),
    ("Boston Common & Public Garden", "Outdoor", 0, 1.5),
    ("Paul Revere House", "Culture", 6, 1.0),
    ("Beer & History Tour", "Food", 55, 2.5),
    ("Kayaking on the Charles River", "Outdoor", 50, 2.0),
    ("Isabella Stewart Gardner Museum", "Culture", 15, 2.0),
    ("Whale Watching from Long Wharf", "Outdoor", 60, 4.0),
  ],
  "New Orleans": [
    ("French Quarter Walking Tour", "Culture", 0, 2.5),
    ("Haunted New Orleans Tour", "Entertainment", 25, 2.0),
    ("Swamp Tour", "Outdoor", 55, 3.0),
    ("Jazz Club Crawl", "Entertainment", 40, 3.0),
    ("WWII Museum", "Culture", 30, 3.5),
    ("Plantation Tour", "Culture", 45, 4.0),
    ("New Orleans Cooking Class", "Food", 90, 3.0),
    ("Mardi Gras Museum", "Culture", 15, 1.5),
    ("Cemetery Tour", "Culture", 20, 1.5),
    ("Riverboat Cruise", "Sightseeing", 35, 2.0),
    ("Bourbon Street Night Tour", "Entertainment", 30, 2.0),
    ("Bayou Kayaking", "Outdoor", 55, 3.0),
    ("Audubon Zoo", "Outdoor", 22, 3.0),
    ("NOLA Food Tour", "Food", 65, 3.0),
    ("New Orleans Art Museum", "Culture", 15, 2.0),
  ],
  "Denver": [
    ("Rocky Mountain National Park Day Trip", "Outdoor", 35, 8.0),
    ("Denver Art Museum", "Culture", 15, 2.5),
    ("Red Rocks Amphitheatre Visit", "Outdoor", 0, 3.0),
    ("Denver Botanic Gardens", "Outdoor", 15, 2.0),
    ("Brewery Tour", "Food", 40, 2.5),
    ("Colorado State Capitol Tour", "Culture", 0, 1.5),
    ("Hiking in Golden", "Outdoor", 0, 4.0),
    ("Denver Food Tour", "Food", 65, 3.0),
    ("Colorado History Museum", "Culture", 12, 2.0),
    ("Whitewater Rafting Day Trip", "Sports", 110, 6.0),
    ("Snowshoeing in Breckenridge", "Sports", 85, 5.0),
    ("Denver Zoo", "Outdoor", 20, 3.0),
    ("16th Street Mall Walk", "Sightseeing", 0, 1.5),
    ("Buffalo Bill Museum", "Culture", 10, 1.5),
    ("Elitch Gardens Theme Park", "Entertainment", 50, 6.0),
  ],
  "Austin": [
    ("Bat Bridge Watch", "Outdoor", 0, 1.5),
    ("6th Street Bar Crawl", "Entertainment", 0, 3.0),
    ("Barton Springs Pool", "Outdoor", 5, 2.0),
    ("Texas State Capitol Tour", "Culture", 0, 1.5),
    ("BBQ Pit Tour", "Food", 55, 3.0),
    ("Kayaking on Lady Bird Lake", "Outdoor", 40, 2.0),
    ("South Congress Avenue Walk", "Sightseeing", 0, 2.0),
    ("Blanton Museum of Art", "Culture", 12, 2.0),
    ("Austin Food & Drink Tour", "Food", 70, 3.0),
    ("Live Music at Antone's", "Entertainment", 20, 2.0),
    ("Barton Creek Greenbelt Hike", "Outdoor", 0, 3.0),
    ("Austin City Limits Music Festival", "Entertainment", 75, 8.0),
    ("LBJ Presidential Library", "Culture", 14, 2.0),
    ("Hamilton Pool Day Trip", "Outdoor", 15, 4.0),
    ("E-scooter Tour of Austin", "Sightseeing", 25, 2.0),
  ],
  "Nashville": [
    ("Country Music Hall of Fame", "Culture", 28, 2.5),
    ("Grand Ole Opry Show", "Entertainment", 75, 3.0),
    ("Honky Tonk Bar Crawl", "Entertainment", 0, 3.0),
    ("Ryman Auditorium Tour", "Culture", 30, 1.5),
    ("Nashville Food Tour", "Food", 65, 3.0),
    ("Parthenon in Centennial Park", "Culture", 10, 1.5),
    ("Belle Meade Historic Site", "Culture", 22, 2.0),
    ("Nashville Zoo", "Outdoor", 18, 3.0),
    ("Mammoth Cave Day Trip", "Outdoor", 55, 6.0),
    ("Tennessee Whiskey Distillery Tour", "Food", 45, 2.0),
    ("Pinewood Social", "Entertainment", 0, 2.0),
    ("Music Row Walking Tour", "Culture", 20, 2.0),
    ("Radnor Lake Hike", "Outdoor", 0, 2.5),
    ("Nashville Sounds Baseball Game", "Sports", 15, 3.0),
    ("Live at the Station Inn", "Entertainment", 15, 2.0),
  ],
}


def _date_range(start: date, end: date) -> list[date]:
  days = (end - start).days
  return [start + timedelta(days=i) for i in range(days + 1)]


def gen_flights() -> pd.DataFrame:
  """Generate flight rows for all city-pair combos across sampled dates."""
  start = date(2026, 6, 1)
  end = date(2026, 12, 31)
  all_dates = _date_range(start, end)

  rows = []
  fid = 1
  for origin in CITIES:
    for dest in CITIES:
      if origin == dest:
        continue
      # Sample ~25 departure dates per route
      sampled = sorted(random.sample(all_dates, min(25, len(all_dates))))
      for dep in sampled:
        # Return date 3–14 days later, capped at end of year
        offset = random.randint(3, 14)
        ret = dep + timedelta(days=offset)
        if ret > end:
          ret = end
        airline = random.choice(AIRLINES)
        price = round(random.uniform(149, 799), 2)
        seats = random.randint(0, 50)
        rows.append({
          "id": f"FL{fid:05d}",
          "origin": origin,
          "destination": dest,
          "depart_date": dep.isoformat(),
          "return_date": ret.isoformat(),
          "airline": airline,
          "price_usd": price,
          "seats_available": seats,
        })
        fid += 1

  return pd.DataFrame(rows)


def gen_hotels() -> pd.DataFrame:
  """Generate hotel rows. check_in/check_out left blank; retrieval fills from state."""
  rows = []
  hid = 1
  for city, names in HOTEL_NAMES.items():
    for name in names:
      n_amenities = random.randint(3, 7)
      amenities = "; ".join(random.sample(AMENITIES_POOL, n_amenities))
      rows.append({
        "id": f"HT{hid:04d}",
        "city": city,
        "name": name,
        "check_in": "",
        "check_out": "",
        "price_per_night_usd": round(random.uniform(89, 449), 2),
        "rating": round(random.uniform(2.5, 5.0), 1),
        "amenities": amenities,
      })
      hid += 1
  return pd.DataFrame(rows)


def gen_restaurants() -> pd.DataFrame:
  """Generate restaurant rows."""
  price_ranges = ["$", "$$", "$$$", "$$$$"]
  rows = []
  rid = 1
  for city, entries in RESTAURANTS.items():
    for name, cuisine in entries:
      rows.append({
        "id": f"RS{rid:04d}",
        "city": city,
        "name": name,
        "cuisine": cuisine,
        "price_range": random.choice(price_ranges),
        "rating": round(random.uniform(3.0, 5.0), 1),
      })
      rid += 1
  return pd.DataFrame(rows)


def gen_activities() -> pd.DataFrame:
  """Generate activity rows."""
  rows = []
  aid = 1
  for city, entries in ACTIVITIES.items():
    for name, category, base_price, duration in entries:
      # Add small random variation to price
      price = round(base_price * random.uniform(0.9, 1.1), 2) if base_price > 0 else 0.0
      rows.append({
        "id": f"AC{aid:04d}",
        "city": city,
        "name": name,
        "category": category,
        "price_usd": price,
        "duration_hours": duration,
      })
      aid += 1
  return pd.DataFrame(rows)


if __name__ == "__main__":
  flights = gen_flights()
  hotels = gen_hotels()
  restaurants = gen_restaurants()
  activities = gen_activities()

  flights.to_csv(OUT / "flights.csv", index=False)
  hotels.to_csv(OUT / "hotels.csv", index=False)
  restaurants.to_csv(OUT / "restaurants.csv", index=False)
  activities.to_csv(OUT / "activities.csv", index=False)

  print(f"flights.csv:     {len(flights):>5} rows")
  print(f"hotels.csv:      {len(hotels):>5} rows")
  print(f"restaurants.csv: {len(restaurants):>5} rows")
  print(f"activities.csv:  {len(activities):>5} rows")
  print("Done.")
