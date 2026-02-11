import networkx as nx
from datetime import datetime

G = nx.Graph()

# Note: Replace these placeholders with your actual information in config.py
# or directly edit them here

G.add_node("User", type="Person", name="YOUR_NAME", surname="YOUR_SURNAME", age='YOUR_AGE', birthday="YOUR_BIRTHDAY")
G.add_node("Favorite Book", type="Book", title="The Lord of the Rings", author="J.R.Tolkien", genre="Fiction")
G.add_node("Favorite Movie", type="Movie", title="Harry Potter", author="J.K.Rowling", genre="Fiction")
G.add_node("Guitar", type="Instrument", name="Guitar")
G.add_edge("User", "Guitar", type="plays", proficiency="intermediate")
G.add_node("Guitar Lessons", type="Event", name="Guitar Lessons", start_date="2020-01-01", end_date="2022-06-30")
G.add_edge("User", "Guitar Lessons", type="attended")
G.add_node("YOUR_LOCATION", type="Location", city="YOUR_LOCATION", country="YOUR_COUNTRY")
G.add_edge("User", "YOUR_LOCATION", type="lives_in")
G.add_edge("User", "Favorite Book", type="has read")
G.add_edge("User", "Favorite Movie", type="has watched")

G.add_node("Piano", type="Instrument", name="Piano")
G.add_edge("User", "Piano", type="plays", proficiency="beginner")
G.add_node("Music Theory", type="Skill", name="Music Theory")
G.add_edge("User", "Music Theory", type="doesn't know")
G.add_node("World War II", type="Event", start_date="1939-09-01", end_date="1945-09-02", description="Global conflict")

G.add_node("Napoleon Bonaparte", type="Person", birthdate="1769-08-15", deathdate="1821-05-05", nationality="French")
G.add_edge("Napoleon Bonaparte", "World War II", type="was not involved in")

G.add_node("Gravity", type="Concept", definition="Force of attraction between objects", discovery_date="1687")
G.add_node("Isaac Newton", type="Person", birthdate="1643-01-04", deathdate="1727-03-31", nationality="English")
G.add_edge("Isaac Newton", "Gravity", type="discovered")

G.add_node("France", type="Country", capital="Paris", population=67000000, language="French")
G.add_node("Italy", type="Country", capital="Rome", population=60300000, language="Italian")
G.add_node("Paris", type="City", country="France", population=2150000, landmarks=["Eiffel Tower", "Louvre Museum"])
G.add_node("Rome", type="City", country="Italy", population=2870000, landmarks=["Colosseum", "Roman Forum"])
G.add_node("Eiffel Tower", type="Landmark", location="Paris", height=324, built_date="1889")
G.add_node("Colosseum", type="Landmark", location="Rome", built_date="80 AD", historical_significance="Ancient Roman amphitheater")

G.add_node("Trip to France", type="Event", start_date="2020-06-01", end_date="2020-06-10", description="Summer vacation in France")
G.add_node("Trip to Italy", type="Event", start_date="2020-07-01", end_date="2020-07-15", description="Summer vacation in Italy")

G.add_node("Plato", type="Person", birthdate="428-427 BCE", deathdate="348-347 BCE", nationality="Ancient Greek")
G.add_node("Aristotle", type="Person", birthdate="384 BCE", deathdate="322 BCE", nationality="Ancient Greek")
G.add_node("Immanuel Kant", type="Person", birthdate="1724 CE", deathdate="1804 CE", nationality="German")
G.add_node("Jean-Paul Sartre", type="Person", birthdate="1905 CE", deathdate="1980 CE", nationality="French")
G.add_node("Martin Heidegger", type="Person", birthdate="1889 CE", deathdate="1976 CE", nationality="German")

G.add_node("Existentialism", type="Concept", definition="Philosophy that emphasizes individual existence and freedom")
G.add_node("Rationalism", type="Concept", definition="Philosophy that emphasizes reason and intellect")
G.add_node("Idealism", type="Concept", definition="Philosophy that emphasizes the mind and spirit over matter")
G.add_node("Ethics", type="Concept", definition="Branch of philosophy that deals with moral principles and values")
G.add_node("Metaphysics", type="Concept", definition="Branch of philosophy that deals with the nature of reality")

G.add_edge("Plato", "Rationalism", type="influenced")
G.add_edge("Aristotle", "Rationalism", type="influenced")
G.add_edge("Immanuel Kant", "Idealism", type="developed")
G.add_edge("Jean-Paul Sartre", "Existentialism", type="developed")
G.add_edge("Martin Heidegger", "Existentialism", type="influenced")
G.add_edge("Plato", "Ethics", type="contributed to")
G.add_edge("Aristotle", "Ethics", type="contributed to")
G.add_edge("Immanuel Kant", "Metaphysics", type="contributed to")

G.add_node("The Republic", type="Book", author="Plato", publication_date="380 BCE")
G.add_node("The Critique of Pure Reason", type="Book", author="Immanuel Kant", publication_date="1781 CE")
G.add_node("Being and Time", type="Book", author="Martin Heidegger", publication_date="1927 CE")
G.add_node("Existentialism is a Humanism", type="Essay", author="Jean-Paul Sartre", publication_date="1946 CE")

G.add_node("Happiness", type="Concept", definition="Positive emotional state")
G.add_node("Freedom", type="Value", definition="Ability to make choices without restriction")

G.add_edge("User", "Happiness", type="values")
G.add_edge("User", "Freedom", type="believes in")
G.add_edge("User", "Existentialism", type="is interested in")

G.add_node("Optimism", type="Trait", definition="Tendency to be hopeful and positive")
G.add_node("Empathy", type="Value", definition="Ability to understand and share feelings of others")

G.add_edge("User", "Optimism", type="possesses")
G.add_edge("User", "Empathy", type="values")

G.add_node("Nihilism", type="Philosophy", definition="Belief that life has no inherent meaning or value")
G.add_edge("User", "Nihilism", type="rejects")

G.add_edge("Plato", "The Republic", type="wrote")
G.add_edge("Immanuel Kant", "The Critique of Pure Reason", type="wrote")
G.add_edge("Martin Heidegger", "Being and Time", type="wrote")
G.add_edge("Jean-Paul Sartre", "Existentialism is a Humanism", type="wrote")

G.add_edge("User", "France", type="visited")
G.add_edge("User", "Italy", type="visited")
G.add_edge("User", "Paris", type="visited")
G.add_edge("User", "Rome", type="visited")
G.add_edge("User", "Eiffel Tower", type="visited")
G.add_edge("User", "Colosseum", type="visited")
G.add_edge("Trip to France", "Paris", type="included")
G.add_edge("Trip to Italy", "Rome", type="included")

G.add_node("Pizza", type="Food", name="Pizza", country="Italian", ingredients=["dough", "tomato sauce", "mozzarella cheese"])
G.add_node("Gyros", type="Food", name="Gyros", country="Greek", ingredients=["Meat", "Potatoes"])
G.add_node("Sushi", type="Food", name="Sushi", country="Japanese", ingredients=["vinegared rice", "raw fish", "seaweed"])

G.add_edge("User", "Pizza", type="loves")
G.add_edge("User", "Gyros", type="loves")
G.add_edge("User", "Sushi", type="dislikes")


def get_person_location(person_name):
    """Get the location where a person lives"""
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            for neighbor in G.neighbors(node):
                neighbor_attributes = G.nodes[neighbor]
                if "type" in neighbor_attributes and neighbor_attributes["type"] == "Location":
                    return neighbor_attributes.get("city")
    return None

def get_favorite_cuisines(person_name):
    """Get favorite cuisines for a person"""
    favorite_foods = []
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            for neighbor in G.neighbors(node):
                neighbor_attributes = G.nodes[neighbor]
                if "type" in neighbor_attributes and neighbor_attributes["type"] == "Food":
                    edge_attributes = G[node][neighbor]
                    if "type" in edge_attributes and edge_attributes["type"] == "loves":
                        favorite_foods.append(neighbor_attributes.get("country") + " " + neighbor_attributes["type"])
    return favorite_foods

def get_favorite_book(person_name):
    """Get favorite book for a person"""
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            for neighbor in G.neighbors(node):
                neighbor_attributes = G.nodes[neighbor]
                if "type" in neighbor_attributes and neighbor_attributes["type"] == "Book":
                    edge_attributes = G[node][neighbor]
                    if "type" in edge_attributes and edge_attributes["type"] == "has read":
                        return neighbor_attributes.get("title"), neighbor_attributes.get("author"), neighbor_attributes.get("genre")
    return None

def get_favorite_food(person_name):
    """Get favorite foods for a person"""
    favorite_foods = []
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            for neighbor in G.neighbors(node):
                neighbor_attributes = G.nodes[neighbor]
                if "type" in neighbor_attributes and neighbor_attributes["type"] == "Food":
                    edge_attributes = G[node][neighbor]
                    if "type" in edge_attributes and edge_attributes["type"] == "loves":
                        favorite_foods.append(neighbor_attributes["name"])
    return favorite_foods

def get_person_age(person_name):
    """Get age of a person"""
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            if "age" in attributes:
                return attributes["age"]
            elif "birthdate" in attributes:
                birthdate = attributes["birthdate"]
                try:
                    birth_year = int(birthdate.split("-")[0])
                    current_year = datetime.now().year
                    return current_year - birth_year
                except ValueError:
                    return None
    return None

def get_person_birthday(person_name):
    """Get birthday of a person"""
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            if "birthday" in attributes:
                return attributes["birthday"]
            elif "birthdate" in attributes:
                return attributes["birthdate"]
    return None

def get_favorite_movie(person_name):
    """Get favorite movie for a person"""
    for node, attributes in G.nodes(data=True):
        if "type" in attributes and attributes["type"] == "Person" and "name" in attributes and attributes["name"] == person_name:
            for neighbor in G.neighbors(node):
                neighbor_attributes = G.nodes[neighbor]
                if "type" in neighbor_attributes and neighbor_attributes["type"] == "Movie":
                    edge_attributes = G[node][neighbor]
                    if "type" in edge_attributes and edge_attributes["type"] == "has watched":
                        return neighbor_attributes["title"]
    return None
