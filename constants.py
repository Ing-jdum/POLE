# Autor: John Urena
# Correo: ing.jdum@gmail.com

queries_dict = {
    'page_rank': '''
        CALL gds.pageRank.stream('social')
        YIELD nodeId, score AS pagerank
        WITH gds.util.asNode(nodeId) AS node, pagerank
        RETURN node.nhs_no AS id, toFloat(pagerank) AS pagerank
        ORDER BY pagerank DESC;
    ''',

    'degree': '''
        CALL gds.degree.stream('social')
        YIELD nodeId, score as degree
        WITH gds.util.asNode(nodeId) AS node, degree
        RETURN node.nhs_no AS id, degree
        ORDER BY degree DESC;
    ''',

    'closeness': '''
        CALL gds.closeness.stream('social')
        YIELD nodeId, score AS closeness
        WITH gds.util.asNode(nodeId) AS node, closeness
        RETURN node.nhs_no AS id, toFloat(closeness) AS closeness
        ORDER BY closeness DESC;
    ''',

    'clustering': '''
        CALL gds.localClusteringCoefficient.stream('social')
        YIELD nodeId, localClusteringCoefficient
        WITH gds.util.asNode(nodeId) AS node, localClusteringCoefficient
        RETURN node.nhs_no AS id, toFloat(localClusteringCoefficient) AS clustering
        ORDER BY clustering DESC;
    ''',

    'betweeness': '''
        CALL gds.betweenness.stream('social')
        YIELD nodeId, score AS centrality
        WITH gds.util.asNode(nodeId) AS node, centrality
        RETURN node.nhs_no AS id, toInteger(centrality) AS betweeness
        ORDER BY betweeness DESC;
    ''',

    'triangle_count': '''
        CALL gds.triangleCount.stream('social')
        YIELD nodeId, triangleCount as triangles
        WITH gds.util.asNode(nodeId) AS node, triangles
        RETURN node.nhs_no AS id, triangles
        ORDER BY triangles DESC;
    ''',

    'commited_crime': '''
        MATCH (person:Person)
        OPTIONAL MATCH (person)-[:PARTY_TO]->(crime:Crime)
        WITH person, COLLECT(crime) AS crimes
        RETURN 
          person.name AS name,
          person.surname AS surname,
          person.nhs_no AS id,
          SIZE(crimes) > 0 AS committed_crime
        ORDER BY id;
    ''',

    'create_social': '''
       CALL gds.graph.project('social',
          'Person',
          {KNOWS: {orientation:'UNDIRECTED'}});
   ''',

    'crime_count': '''
        MATCH (person:Person)
        OPTIONAL MATCH (person)-[:PARTY_TO]->(crime:Crime)
        RETURN 
          person.name AS name,
          person.surname AS surname,
          person.nhs_no AS id,
          count(crime) AS crime_count;
    ''',

    'length_to_criminal': '''
        MATCH (p:Person), (cp:Person)-[:PARTY_TO]->(c:Crime)
        WHERE p <> cp
        WITH p, cp, shortestPath((p)-[:KNOWS*..]-(cp)) AS path, c
        WITH p, cp, length(path) AS path_length, COLLECT(DISTINCT c) AS crimes
        ORDER BY p.nhs_no, path_length
        WITH p, COLLECT(path_length)[0] AS closest_crime_path_length
        RETURN p.nhs_no AS id, min(closest_crime_path_length) AS nearest_path;
    ''',

    'known_criminals_3': '''
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:KNOWS*..3]->(criminal:Person)-[:PARTY_TO]->(c:Crime)
        WITH p.nhs_no AS person_nhs_no, criminal, COLLECT(DISTINCT c) AS crimes
        RETURN person_nhs_no AS id, COUNT(DISTINCT criminal) AS connected_criminals_count, 
        SUM(SIZE(crimes)) AS total_crimes_count;
    ''',

    'lives_with_criminal': '''
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:KNOWS_LW]-(criminal:Person)-[:PARTY_TO]-(c:Crime)
        RETURN distinct(p.nhs_no) AS id, CASE WHEN criminal IS NOT NULL THEN 1 ELSE 0 END AS lives_with_criminal
    '''
}
