import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

// REPLACE WITH YOUR COMPUTER'S LAN IP (e.g., http://192.168.1.5:8000)
// For Android Emulator, use 'http://10.0.2.2:8000'
const API_URL = 'http://192.168.1.6:8000';

const api = axios.create({
    baseURL: API_URL,
    timeout: 10000, // 10 seconds
    headers: {
        'Content-Type': 'application/json',
    },
});

let onUnauthorizedCallback = null;

export const setUnauthorizedCallback = (callback) => {
    console.log('api.js: setUnauthorizedCallback called');
    onUnauthorizedCallback = callback;
};

export const setAuthToken = (token) => {
    if (token) {
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        console.log('api.js: Auth token set');
    } else {
        delete api.defaults.headers.common['Authorization'];
        console.log('api.js: Auth token cleared');
    }
};

api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        if (error.response && error.response.status === 401 && !originalRequest._retry && !originalRequest.url.includes('/auth/refresh')) {
            console.log('api.js: 401 detected, attempting refresh');
            originalRequest._retry = true;
            try {
                const refreshToken = await SecureStore.getItemAsync('refreshToken');
                if (refreshToken) {
                    console.log('api.js: Refresh token found, calling endpoint');
                    const { access_token, refresh_token: newRefreshToken } = await refreshTokenCall(refreshToken);
                    await SecureStore.setItemAsync('userToken', access_token);
                    if (newRefreshToken) {
                        await SecureStore.setItemAsync('refreshToken', newRefreshToken);
                    }
                    api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
                    originalRequest.headers['Authorization'] = `Bearer ${access_token}`;
                    return api(originalRequest);
                } else {
                    console.log('api.js: No refresh token found');
                    throw new Error('No refresh token available');
                }
            } catch (refreshError) {
                console.error('api.js: Token refresh failed:', refreshError);
                await SecureStore.deleteItemAsync('userToken');
                await SecureStore.deleteItemAsync('refreshToken');
                if (onUnauthorizedCallback) {
                    console.log('api.js: Calling onUnauthorizedCallback');
                    onUnauthorizedCallback();
                } else {
                    console.error('api.js: onUnauthorizedCallback is null!');
                }
            }
        }
        return Promise.reject(error);
    }
);

export const login = async (username, password) => {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post('/auth/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return response.data;
};

export const register = async (username, password) => {
    const response = await api.post('/auth/register', { username, password });
    return response.data;
};

export const refreshTokenCall = async (token) => {
    const response = await api.post('/auth/refresh', null, {
        params: { refresh_token: token }
    });
    return response.data;
};

export const getFeed = async (type) => {
    const params = type ? { item_type: type } : {};
    const response = await api.get('/feed/', { params });
    return response.data;
};

export const getMatch = async (matchType) => {
    const response = await api.get('/feed/match', { params: { match_type: matchType } });
    return response.data;
};

export const swipeItem = async (itemId, action) => {
    const response = await api.post('/swipe/', { item_id: itemId, action });
    return response.data;
};

export const getProfile = async () => {
    const response = await api.get('/auth/profile');
    return response.data;
};

export const getDailyFeed = async () => {
    const response = await api.get('/feed/daily');
    return response.data;
};

export const updateProfile = async (data) => {
    const response = await api.patch('/auth/profile', data);
    return response.data;
};

export const getSocialMatches = async () => {
    try {
        const response = await api.get('/social/candidates');
        return response.data;
    } catch (error) {
        console.error("Error fetching matches:", error);
        throw error;
    }
};

export const swipeUser = async (likedUserId, action) => {
    try {
        const response = await api.post('/social/swipe', {
            liked_user_id: likedUserId,
            action: action
        });
        return response.data;
    } catch (error) {
        console.error("Error swiping user:", error);
        throw error;
    }
};

export const refreshUserVector = async () => {
    const response = await api.post('/social/refresh_vector');
    return response.data;
};

export const getConfirmedMatches = async () => {
    try {
        const response = await api.get('/social/confirmed_matches');
        return response.data;
    } catch (error) {
        console.error("Error fetching confirmed matches:", error);
        throw error;
    }
};

export const getUserProfilePublic = async (userId) => {
    const response = await api.get(`/users/${userId}/profile`);
    return response.data;
};

export const blockUser = async (userId) => {
    const response = await api.post(`/users/${userId}/block`);
    return response.data;
};

export const reportUser = async (userId, reason, details) => {
    const response = await api.post(`/users/${userId}/report`, { reason, details });
    return response.data;
};

// Friends API
export const getFriends = async () => {
    const response = await api.get('/friends/list');
    return response.data;
};

export const getFriendRequests = async () => {
    const response = await api.get('/friends/requests');
    return response.data;
};

export const sendFriendRequest = async (userId) => {
    const response = await api.post('/friends/request', { receiver_id: userId });
    return response.data;
};

export const acceptFriendRequest = async (requestId) => {
    const response = await api.post(`/friends/accept/${requestId}`);
    return response.data;
};

export const rejectFriendRequest = async (requestId) => {
    const response = await api.post(`/friends/reject/${requestId}`);
    return response.data;
};

export const getDateRecommendations = async (partnerId) => {
    const response = await api.get('/date/recommendations', { params: { partner_id: partnerId } });
    return response.data;
};

export default api;
