from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool #download websites process and create vector db to search it
#from crewai_tools import FileReadTool, FileWriterTool

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Cody_Crew_Juris():
	"""Cody_Crew_Juris crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	llm_ = LLM(
			model="ollama/llama3.2",
			base_url="http://localhost:11434"
		)

	# def __init__(self):
	# 	self.llm_ = LLM(
	# 		model="ollama/llama3.2",
	# 		base_url="http://localhost:11434"
	# 	)

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def juris_researcher(self) -> Agent:

		tool = SerperDevTool(
			country="br",
			locale="br",
			location="Brasilia, Distrito Federal, Brasil",
			n_results=15,
			)

		return Agent(
			config=self.agents_config['juris_researcher'],
			verbose=True,
			tools=[tool, ScrapeWebsiteTool()],
			llm=self.llm_
		)

	@agent
	def reporting_analyst(self) -> Agent:

		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True,
			tools=[SerperDevTool(), WebsiteSearchTool()],
			llm=self.llm_
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			context=[self.research_task()],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the especialista_em_jurisprudencia crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
