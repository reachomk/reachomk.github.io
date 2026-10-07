(() => {
  const form = document.getElementById('hero-challenge');
  const feedback = document.getElementById('hero-feedback-box');
  const submit = document.getElementById('hero-submit-button');
  const choices = Array.from(form.elements.answer);

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const selected = choices.find((choice) => choice.checked);
    if (!selected) {
      feedback.textContent = 'Choose a letter before checking your answer.';
      feedback.className = 'challenge-feedback is-incorrect';
      choices[0].focus();
      return;
    }
    const correct = selected.value === 'D';
    feedback.textContent = correct
      ? 'Correct! D, the sofa, has an orientation that does not match the change in viewpoint.'
      : `You chose ${selected.value}. The answer is D, the sofa: its orientation does not match the change in viewpoint.`;
    feedback.className = `challenge-feedback ${correct ? 'is-correct' : 'is-incorrect'}`;
    choices.forEach((choice) => {
      choice.disabled = true;
      choice.closest('label').classList.toggle('is-correct', choice.value === 'D');
    });
    submit.disabled = true;
  });

  form.addEventListener('reset', () => {
    choices.forEach((choice) => {
      choice.disabled = false;
      choice.closest('label').classList.remove('is-correct');
    });
    submit.disabled = false;
    feedback.className = 'challenge-feedback';
    feedback.textContent = 'Select a letter to check your answer.';
  });
})();
