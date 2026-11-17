from dotenv import load_dotenv

load_dotenv()  # load SERPER_API_KEY, MODEL, LLM key from .env before importing crew

from travel_planner.crew import TravelPlannerCrew  # noqa: E402


def main():
    print("=== AI Travel Planner (CrewAI) ===")
    inputs = {
        "origin": input("Where are you travelling from? ") or "Chennai",
        "city": input("Which city do you want to visit? ") or "Tokyo",
        "travel_dates": input("Travel dates (e.g. 10-15 Dec 2026)? ") or "10-15 Dec 2026",
        "interests": input("Your interests (food, history, ...)? ") or "food, culture",
    }

    result = TravelPlannerCrew().crew().kickoff(inputs=inputs)

    print("\n\n=== YOUR TRAVEL PLAN ===\n")
    print(result.raw)
    print("\nSaved to output/travel_plan.md")


if __name__ == "__main__":
    main()
