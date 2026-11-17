import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from travel_planner.tools import scrape_tool, search_tool


@CrewBase
class TravelPlannerCrew:
    """Three agents working in sequence: research -> local guide -> itinerary."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        # Local Ollama model, e.g. MODEL=ollama/qwen2.5:3b
        self.llm = LLM(
            model=os.getenv("MODEL", "ollama/qwen2.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # ---------- Agents ----------
    @agent
    def city_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["city_researcher"],
            tools=[search_tool, scrape_tool],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def local_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["local_expert"],
            tools=[search_tool, scrape_tool],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def travel_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["travel_planner"],
            llm=self.llm,
            verbose=True,
        )

    # ---------- Tasks ----------
    @task
    def research_city_task(self) -> Task:
        return Task(config=self.tasks_config["research_city_task"])

    @task
    def local_guide_task(self) -> Task:
        return Task(config=self.tasks_config["local_guide_task"])

    @task
    def itinerary_task(self) -> Task:
        return Task(config=self.tasks_config["itinerary_task"])

    # ---------- Crew ----------
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
