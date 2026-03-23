import openreview
import json

# Initialize the OpenReview client
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username='yingxuan.wen@edu.em-lyon.com',
    password='Amanda2310504'
)


def get_active_submissions(client, venue_id):
    # Get the submission invitation name
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content['submission_name']['value']

    # Fetch all submissions with details including replies (such as reviews)
    all_submissions = client.get_all_notes(invitation=f'{venue_id}/-/{submission_name}', details='replies')

    # Filter out only active submissions
    active_submissions = [submission for submission in all_submissions if submission.content.get('venueid') == venue_id]

    return active_submissions

# Define the conference venue ID
venue_id = 'ICLR.cc/2026/Conference'
review_name = 'Official_Review'  # Replace with the actual review invitation name if different
decision_name = 'Decision'  # Replace if the actual decision invitation name differs
meta_review_name = 'Meta_Review'  # Replace if the actual meta review invitation name differs

# Fetch and print active submissions along with their reviews, decision, and meta review
venue_group = client.get_group(venue_id)
submission_name = venue_group.content['submission_name']['value']

# Fetch all submissions with details including replies (such as reviews)
all_submissions = client.get_all_notes(invitation=f'{venue_id}/-/{submission_name}', details='replies')
# active_submissions = [
#         submission for submission in all_submissions
#         if (submission.content.get('venueid') == venue_id) or
#            (isinstance(submission.content.get('venueid'), dict) and submission.content['venueid'].get('value') == venue_id)
#     ]
print(len(all_submissions))
file_name = f"{venue_id.split('/')[0]}_{venue_id.split('/')[1]}.jsonl"
with open(file_name, 'w', encoding='utf-8') as f:
    for submission in all_submissions:
        review_list = []
        for reply in submission.details['replies']:
            review = openreview.api.Note.from_json(reply)
            review = review.to_json()
            review_list.append(review)
        sub_json = submission.to_json()
        sub_json['reviews'] = review_list

        json_line = json.dumps(sub_json)

        f.write(json_line + '\n')
        f.flush()