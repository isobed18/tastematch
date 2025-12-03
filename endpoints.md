# Backend API Documentation

Base URL: `http://localhost:8000`

## Authentication

### Register User
**POST** `/auth/register`
*   **Description:** Creates a new user account.
*   **Body:**
    ```json
    {
      "username": "user1",
      "password": "securepassword"
    }
    ```
*   **Response:** `200 OK` (User created)

### Login
**POST** `/auth/login`
*   **Description:** Authenticates user and returns access/refresh tokens.
*   **Content-Type:** `application/x-www-form-urlencoded`
*   **Form Data:**
    *   `username`: "user1"
    *   `password`: "securepassword"
*   **Response:**
    ```json
    {
      "access_token": "eyJhbG...",
      "refresh_token": "eyJxdW...",
      "token_type": "bearer",
      "username": "user1",
      "user_id": 1
    }
    ```

### Refresh Token
**POST** `/auth/refresh`
*   **Description:** Get a new access token using a refresh token.
*   **Query Params:** `refresh_token=YOUR_REFRESH_TOKEN`
*   **Response:** New tokens.

### Get Profile
**GET** `/auth/profile`
*   **Headers:** `Authorization: Bearer <access_token>`
*   **Response:** User details.

---

## Feed & Recommendations

### Get Feed
**GET** `/feed/`
*   **Description:** Returns a mixed list of recommendations (personalized + exploration).
*   **Headers:** `Authorization: Bearer <access_token>`
*   **Query Params:**
    *   `limit`: Number of items (default: 20)
*   **Response:** List of `ItemOut` objects.
    ```json
    [
      {
        "id": 101,
        "title": "Inception",
        "image_url": "https://image.tmdb.org/...",
        "is_recommendation": true,
        "match_type": "perfect",
        "match_score": 0.98
      }
    ]
    ```

### Get Specific Match
**GET** `/feed/match`
*   **Description:** Returns a single item of a specific match type (triggered by frontend logic).
*   **Headers:** `Authorization: Bearer <access_token>`
*   **Query Params:**
    *   `match_type`: "perfect" or "reverse"
*   **Response:** Single `ItemOut` object.

---

## Interactions

### Swipe Item
**POST** `/swipe/`
*   **Description:** Record a user interaction (swipe).
*   **Headers:** `Authorization: Bearer <access_token>`
*   **Body:**
    ```json
    {
      "item_id": 101,
      "action": "like" 
    }
    ```
    *   *Actions:* "like", "dislike", "superlike", "watchlist"
*   **Response:** `201 Created`
