from neo4j import GraphDatabase
uri = "neo4j+s://4e9b2d24.databases.neo4j.io"
driver = GraphDatabase.driver(uri, auth=("neo4j", "4MyZedIjW_ZdPpR7xzXx5ocCENECin_jZh6WThHHktU"))
with driver.session() as session:
    result = session.run("MATCH (n) RETURN n LIMIT 5")
    print([record.data() for record in result])