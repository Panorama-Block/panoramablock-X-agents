from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Agents():
	"""Agents crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@agent
	def twitter_redactor(self) -> Agent:
		return Agent(
			config=self.agents_config['twitter_redactor'],
			verbose=True
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
	def report_task(self) -> Task:
		return Task(
			config=self.tasks_config['report_task'],
			context=[self.research_task],
			output_file='report.md'
		)

	@task
	def twitter_redaction_task(self) -> Task:
		return Task(
			config=self.tasks_config['twitter_redaction_task'],
			context=[self.report_task],
			output_file='tweet.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Agents crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=[
				self.researcher,
				self.reporting_analyst,
				self.twitter_redactor
			],
			tasks=[
				self.research_task,
				self.report_task,
				self.twitter_redaction_task
			],
			verbose=True
		)
