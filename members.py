def get_team_members_details() -> dict:
    """
    Returns a dictionary containing details of team members.

    Each team member is represented as a dictionary with the following keys:
    - name: The name of the team member.
    - description: A brief description of the team member's role and responsibilities.

    Returns:
    A dictionary containing details of team members.
    """
    members_dict = [
        {
            "name": "ResumeAnalyzer",
            "description": "Responsible for analyzing resumes to extract key information.",
        },
        {
            "name": "CoverLetterGenerator",
            "description": "Specializes in creating and optimizing cover letters tailored to job descriptions. Highlights the candidate's strengths and ensures the cover letter aligns with the requirements of the position.",
        },
        {
            "name": "JobSearcher",
            "description": "Conducts job searches based on specified criteria such as industry, location, and job title.",
        },
        {
            "name": "WebResearcher",
            "description": "Answers technical learning and interview questions using the local Hello-Agents knowledge base; uses web search for current company, industry and news information.",
        },
        {
            "name": "ChatBot",
            "description": "Handles greetings, thanks, closing replies, simple summaries of existing messages, and clarification of ambiguous or unsupported task combinations."
        },
    ]
    return members_dict
