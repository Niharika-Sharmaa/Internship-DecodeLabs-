# DecodeLabs Internship - Project 3
# AI Recommendation Logic
#
# Requirements covered:
# 1. Take user input (choices/interests)
# 2. Match preferences using logic/similarity
# 3. Display recommended items
# 4. Demonstrate logic building, pattern matching,
#    and recommendation concepts
# 5. Calculate similarity scores between user preferences
#    and item attributes
# 6. Allow the user to rate preferences

import math


# ---------------------------------------------------------
# RECOMMENDATION DATA
# ---------------------------------------------------------

items = [
    {
        "name": "Inception",
        "category": "Movie",
        "genres": {"sci-fi", "thriller", "action"},
        "tags": {"mind-bending", "technology", "dream", "mystery"}
    },
    {
        "name": "Interstellar",
        "category": "Movie",
        "genres": {"sci-fi", "drama", "adventure"},
        "tags": {"space", "science", "technology", "emotional"}
    },
    {
        "name": "The Dark Knight",
        "category": "Movie",
        "genres": {"action", "thriller", "crime"},
        "tags": {"superhero", "crime", "justice", "dark"}
    },
    {
        "name": "Avengers: Endgame",
        "category": "Movie",
        "genres": {"action", "adventure", "sci-fi"},
        "tags": {"superhero", "technology", "team", "battle"}
    },
    {
        "name": "The Matrix",
        "category": "Movie",
        "genres": {"sci-fi", "action", "thriller"},
        "tags": {"technology", "artificial-intelligence", "virtual", "mind-bending"}
    },
    {
        "name": "The Social Network",
        "category": "Movie",
        "genres": {"drama", "biography"},
        "tags": {"technology", "business", "programming", "entrepreneurship"}
    },
    {
        "name": "The Martian",
        "category": "Movie",
        "genres": {"sci-fi", "adventure", "drama"},
        "tags": {"space", "science", "survival", "technology"}
    },
    {
        "name": "Sherlock",
        "category": "Series",
        "genres": {"mystery", "crime", "drama"},
        "tags": {"detective", "investigation", "intelligence", "suspense"}
    },
    {
        "name": "Stranger Things",
        "category": "Series",
        "genres": {"sci-fi", "horror", "mystery"},
        "tags": {"supernatural", "friendship", "mystery", "adventure"}
    },
    {
        "name": "Breaking Bad",
        "category": "Series",
        "genres": {"crime", "drama", "thriller"},
        "tags": {"crime", "chemistry", "intelligence", "suspense"}
    },
    {
        "name": "Money Heist",
        "category": "Series",
        "genres": {"crime", "thriller", "drama"},
        "tags": {"strategy", "team", "intelligence", "suspense"}
    },
    {
        "name": "Black Mirror",
        "category": "Series",
        "genres": {"sci-fi", "drama", "thriller"},
        "tags": {"technology", "artificial-intelligence", "future", "dark"}
    }
]


# ---------------------------------------------------------
# USER PREFERENCE INPUT
# ---------------------------------------------------------

available_genres = sorted(
    {genre for item in items for genre in item["genres"]}
)

available_tags = sorted(
    {tag for item in items for tag in item["tags"]}
)


def display_options():
    print("\nAvailable genres:")
    print(", ".join(available_genres))

    print("\nAvailable interests:")
    print(", ".join(available_tags))


def get_user_preferences():
    print("\n" + "=" * 65)
    print("              USER PREFERENCE INPUT")
    print("=" * 65)

    display_options()

    genre_input = input(
        "\nEnter your preferred genres (comma-separated): "
    ).strip().lower()

    interest_input = input(
        "Enter your interests (comma-separated): "
    ).strip().lower()

    genres = {
        value.strip()
        for value in genre_input.split(",")
        if value.strip()
    }

    interests = {
        value.strip()
        for value in interest_input.split(",")
        if value.strip()
    }

    return genres, interests


# ---------------------------------------------------------
# SIMILARITY CALCULATION
# ---------------------------------------------------------

def calculate_similarity(user_genres, user_interests, item):
    """
    Calculates a similarity score between the user's
    preferences and an item's attributes.

    Genre similarity and interest/tag similarity are
    calculated separately and then combined.
    """

    item_genres = item["genres"]
    item_tags = item["tags"]

    matched_genres = user_genres.intersection(item_genres)
    matched_interests = user_interests.intersection(item_tags)

    # Jaccard-style similarity
    genre_union = user_genres.union(item_genres)
    interest_union = user_interests.union(item_tags)

    genre_score = (
        len(matched_genres) / len(genre_union)
        if genre_union
        else 0
    )

    interest_score = (
        len(matched_interests) / len(interest_union)
        if interest_union
        else 0
    )

    # Give genres slightly more importance than tags.
    final_score = (0.6 * genre_score) + (0.4 * interest_score)

    return (
        final_score,
        matched_genres,
        matched_interests
    )


# ---------------------------------------------------------
# RATING SYSTEM
# ---------------------------------------------------------

def collect_ratings(recommendations):
    """
    Allows the user to rate recommended items.
    Ratings are stored for the current session.
    """

    ratings = {}

    print("\n" + "=" * 65)
    print("                    RATE ITEMS")
    print("=" * 65)

    print("Rate each recommendation from 1 to 5.")
    print("Enter 0 if you do not want to rate an item.")

    for item in recommendations:
        while True:
            try:
                rating = int(
                    input(f"\nRate '{item['name']}' (1-5, 0 to skip): ")
                )

                if 0 <= rating <= 5:
                    ratings[item["name"]] = rating
                    break

                print("Please enter a number between 0 and 5.")

            except ValueError:
                print("Please enter a valid number.")

    return ratings


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------

def generate_recommendations(user_genres, user_interests):
    recommendations = []

    for item in items:
        score, matched_genres, matched_interests = calculate_similarity(
            user_genres,
            user_interests,
            item
        )

        recommendations.append({
            "name": item["name"],
            "category": item["category"],
            "score": score,
            "matched_genres": matched_genres,
            "matched_interests": matched_interests
        })

    recommendations.sort(
        key=lambda recommendation: recommendation["score"],
        reverse=True
    )

    return recommendations


# ---------------------------------------------------------
# DISPLAY RECOMMENDATIONS
# ---------------------------------------------------------

def display_recommendations(recommendations, limit=5):
    print("\n" + "=" * 65)
    print("                 RECOMMENDED ITEMS")
    print("=" * 65)

    displayed = recommendations[:limit]

    if not displayed:
        print("No recommendations found.")
        return

    for index, item in enumerate(displayed, start=1):
        percentage = item["score"] * 100

        print(f"\n{index}. {item['name']}")
        print(f"   Type: {item['category']}")
        print(f"   Similarity Score: {percentage:.2f}%")

        if item["matched_genres"]:
            print(
                "   Matching Genres: "
                + ", ".join(sorted(item["matched_genres"]))
            )

        if item["matched_interests"]:
            print(
                "   Matching Interests: "
                + ", ".join(sorted(item["matched_interests"]))
            )


# ---------------------------------------------------------
# RATING-BASED RE-RANKING
# ---------------------------------------------------------

def rerank_using_ratings(recommendations, ratings):
    """
    Uses user ratings to demonstrate preference refinement.

    Higher-rated items receive a small ranking boost.
    """

    for item in recommendations:
        rating = ratings.get(item["name"], 0)

        # Rating contributes up to 20% additional score.
        rating_boost = (rating / 5) * 0.20

        item["personalized_score"] = (
            item["score"] + rating_boost
        )

    recommendations.sort(
        key=lambda recommendation: recommendation["personalized_score"],
        reverse=True
    )

    return recommendations


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("          DECODELABS - PROJECT 3")
    print("             AI RECOMMENDATION LOGIC")
    print("=" * 65)

    print("\nGoal:")
    print("Create recommendations based on user preferences.")
    print("The system uses pattern matching and similarity logic")
    print("instead of random suggestions.")

    user_genres, user_interests = get_user_preferences()

    if not user_genres and not user_interests:
        print("\nNo preferences were entered.")
        print("Please enter at least one genre or interest.")
        return

    recommendations = generate_recommendations(
        user_genres,
        user_interests
    )

    display_recommendations(recommendations)

    choice = input(
        "\nWould you like to rate the recommendations? (yes/no): "
    ).strip().lower()

    if choice in {"yes", "y"}:
        ratings = collect_ratings(recommendations[:5])

        recommendations = rerank_using_ratings(
            recommendations,
            ratings
        )

        print("\n" + "=" * 65)
        print("          PERSONALIZED RECOMMENDATIONS")
        print("=" * 65)

        for index, item in enumerate(recommendations[:5], start=1):
            print(
                f"{index}. {item['name']} "
                f"- Personalized Score: "
                f"{item['personalized_score'] * 100:.2f}%"
            )

    print("\n" + "=" * 65)
    print("Recommendation process completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
